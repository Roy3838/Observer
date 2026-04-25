// tauri-plugin-llm-engine - Native LLM inference engine using llama.cpp with Metal acceleration
// This plugin provides streaming text generation from GGUF models.
// Supports multimodal (vision) models via mmproj (multimodal projector) files.

use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

// ── Log emitter ──────────────────────────────────────────────────────────────
// The app sets this once at startup via `set_log_emitter`. Every `llm_log!`
// call invokes it so log lines cross the IPC bridge into the JS Logger.

static LOG_EMITTER: OnceLock<Box<dyn Fn(&str, &str) + Send + Sync>> = OnceLock::new();

/// Register a function that forwards log lines to the frontend.
/// Call this once during Tauri setup before any engine activity.
/// `f(level, message)` — level is one of "info" | "warn" | "error".
pub fn set_log_emitter(f: impl Fn(&str, &str) + Send + Sync + 'static) {
    let _ = LOG_EMITTER.set(Box::new(f));
}

/// Internal logging helper — emits via the registered emitter when available,
/// always falls back to eprintln so logs are never silently dropped.
#[doc(hidden)]
pub fn _emit_log(level: &str, message: String) {
    if let Some(emit) = LOG_EMITTER.get() {
        emit(level, &message);
    } else {
        eprintln!("[LlmEngine][{}] {}", level, message);
    }
}

/// Structured log macro that routes through the registered emitter.
/// Falls back to eprintln when no emitter is set (e.g. in unit tests).
#[macro_export]
macro_rules! llm_log {
    (warn, $($arg:tt)*) => { $crate::_emit_log("warn",  format!($($arg)*)) };
    (error, $($arg:tt)*) => { $crate::_emit_log("error", format!($($arg)*)) };
    ($($arg:tt)*)        => { $crate::_emit_log("info",  format!($($arg)*)) };
}

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel, Special};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::mtmd::{MtmdContext, MtmdContextParams, MtmdBitmap, MtmdInputText, mtmd_default_marker};

/// Generation metrics for debugging and performance monitoring
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerationMetrics {
    pub tokens_generated: u32,
    pub prompt_tokens: u32,
    pub time_to_first_token_ms: f64,
    pub total_generation_time_ms: f64,
    pub tokens_per_second: f64,
}

/// Configurable sampler parameters for text generation
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SamplerParams {
    pub temperature: f32,      // 0.0-2.0, default 0.7
    pub top_p: f32,            // 0.0-1.0, default 0.9
    pub top_k: i32,            // 0=disabled, default 40
    pub seed: u32,             // default 42
    pub repeat_penalty: f32,   // 1.0-2.0, default 1.1
}

impl Default for SamplerParams {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_p: 0.9,
            top_k: 64,
            seed: 42,
            repeat_penalty: 1.0,
        }
    }
}

/// Metadata for a downloaded GGUF model
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NativeModelInfo {
    pub id: String,           // Unique identifier (filename without extension)
    pub name: String,         // Display name
    pub filename: String,     // Actual filename on disk
    pub size_bytes: u64,      // File size
    pub hf_url: Option<String>, // Original HuggingFace URL if known
    pub mmproj_filename: Option<String>, // Associated mmproj file if multimodal
    pub is_multimodal: bool,  // Whether model supports vision/multimodal
}

/// Content part for multimodal messages
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(tag = "type")]
pub enum ChatContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { image: String }, // base64 encoded image data (data URL or raw base64)
}

/// Message content - either a simple string or multimodal parts
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(untagged)]
pub enum ChatContent {
    Text(String),
    Parts(Vec<ChatContentPart>),
}

/// Message format for chat completion with multimodal support
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: ChatContent,
}

/// LLM Engine state - holds the loaded model and context
pub struct LlmEngine {
    backend: LlamaBackend,
    model: Option<LlamaModel>,
    model_path: Option<PathBuf>,
    loaded_model_id: Option<String>,
    mtmd_ctx: Option<MtmdContext>, // Multimodal context (if mmproj loaded)
    mmproj_path: Option<PathBuf>,  // Path to loaded mmproj file
    sampler_params: SamplerParams, // Configurable sampler parameters
    last_metrics: Option<GenerationMetrics>, // Metrics from last generation
    use_gpu: bool, // Whether to use GPU acceleration (Metal). Default false for compatibility.
}

impl LlmEngine {
    /// Create a new uninitialized engine
    pub fn new() -> Result<Self, String> {
        let backend = LlamaBackend::init()
            .map_err(|e| format!("Failed to init llama backend: {}", e))?;

        llm_log!("Backend initialized");
        Ok(Self {
            backend,
            model: None,
            model_path: None,
            loaded_model_id: None,
            mtmd_ctx: None,
            mmproj_path: None,
            sampler_params: SamplerParams::default(),
            last_metrics: None,
            use_gpu: false, // Default to CPU for maximum compatibility
        })
    }

    /// Load a GGUF model from the given path.
    /// Loads a model from `model_path`. If `mmproj_path` is provided, the multimodal
    /// projector is loaded alongside it; otherwise the model is treated as text-only.
    pub fn load_model(&mut self, model_path: PathBuf, model_id: String, mmproj_path: Option<PathBuf>) -> Result<(), String> {
        llm_log!("Loading model: {:?}", model_path);

        if !model_path.exists() {
            return Err(format!("Model file not found: {:?}", model_path));
        }

        self.unload();

        // Use GPU layers based on setting: 99 for GPU (offload all), 0 for CPU only
        let n_gpu_layers = if self.use_gpu { 99 } else { 0 };
        llm_log!("GPU mode: {}, n_gpu_layers: {}", self.use_gpu, n_gpu_layers);

        let model_params = LlamaModelParams::default()
            .with_n_gpu_layers(n_gpu_layers);

        let model = LlamaModel::load_from_file(&self.backend, &model_path, &model_params)
            .map_err(|e| format!("Failed to load model: {}", e))?;

        self.model = Some(model);
        self.model_path = Some(model_path.clone());
        self.loaded_model_id = Some(model_id.clone());

        if let Some(mmproj) = mmproj_path {
            if let Err(e) = self.load_mmproj(mmproj) {
                llm_log!(warn, "Failed to load mmproj: {}", e);
            }
        }

        llm_log!("Model loaded: {} (multimodal: {})", model_id, self.is_multimodal());
        Ok(())
    }

    /// Load a multimodal projector (mmproj) file for vision support
    pub fn load_mmproj(&mut self, mmproj_path: PathBuf) -> Result<(), String> {
        if !mmproj_path.exists() {
            return Err(format!("mmproj file not found: {:?}", mmproj_path));
        }

        let model = self.model.as_ref().ok_or("No model loaded")?;

        // Create mtmd context params with GPU acceleration (Metal)
        let mut mtmd_params = MtmdContextParams::default();
        mtmd_params.use_gpu = self.use_gpu;

        let mmproj_str = mmproj_path.to_str()
            .ok_or_else(|| "Invalid mmproj path".to_string())?;

        let mtmd_ctx = MtmdContext::init_from_file(mmproj_str, model, &mtmd_params)
            .map_err(|e| format!("Failed to create mtmd context: {:?}", e))?;

        llm_log!("mmproj loaded: {:?}", mmproj_path);
        self.mtmd_ctx = Some(mtmd_ctx);
        self.mmproj_path = Some(mmproj_path);

        Ok(())
    }

    /// Check if multimodal (vision) is available
    pub fn is_multimodal(&self) -> bool {
        self.mtmd_ctx.is_some()
    }

    /// Unload the current model and free memory
    pub fn unload(&mut self) {
        if self.model.is_some() {
            llm_log!("Unloading model");
            // IMPORTANT: Drop mtmd_ctx first since it holds a reference to the model
            self.mtmd_ctx = None;
            self.mmproj_path = None;
            self.model = None;
            self.model_path = None;
            self.loaded_model_id = None;
        }
    }

    /// Check if a model is loaded
    pub fn is_loaded(&self) -> bool {
        self.model.is_some()
    }

    /// Get the ID of the currently loaded model
    pub fn loaded_model_id(&self) -> Option<&String> {
        self.loaded_model_id.as_ref()
    }

    /// Get the current sampler parameters
    pub fn get_sampler_params(&self) -> &SamplerParams {
        &self.sampler_params
    }

    /// Set the sampler parameters
    pub fn set_sampler_params(&mut self, params: SamplerParams) {
        self.sampler_params = params;
    }

    /// Get whether GPU acceleration is enabled
    pub fn get_use_gpu(&self) -> bool {
        self.use_gpu
    }

    /// Set whether to use GPU acceleration (Metal)
    /// Must be called before load_model to take effect
    pub fn set_use_gpu(&mut self, use_gpu: bool) {
        self.use_gpu = use_gpu;
        llm_log!("GPU mode set to: {}", use_gpu);
    }

    /// Get the last generation metrics
    pub fn get_last_metrics(&self) -> Option<&GenerationMetrics> {
        self.last_metrics.as_ref()
    }

    /// Get the path to the loaded model
    pub fn get_model_path(&self) -> Option<&PathBuf> {
        self.model_path.as_ref()
    }

    /// Get the path to the loaded mmproj
    pub fn get_mmproj_path(&self) -> Option<&PathBuf> {
        self.mmproj_path.as_ref()
    }

    /// Generate a response from chat messages with streaming callback
    /// Supports both text-only and multimodal (image + text) messages
    pub fn generate<F>(
        &mut self,
        messages: Vec<ChatMessage>,
        mut on_token: F,
    ) -> Result<String, String>
    where
        F: FnMut(&str) -> bool, // return false to cancel generation
    {
        let model = self.model.as_ref().ok_or("No model loaded")?;

        // Extract images from messages for multimodal processing
        let images = self.extract_images(&messages);
        let has_images = !images.is_empty();

        if has_images && self.mtmd_ctx.is_none() {
            llm_log!(warn, "Images provided but no mmproj loaded");
        }

        // If we have images and multimodal context, use mtmd pipeline
        if has_images && self.mtmd_ctx.is_some() {
            return self.generate_multimodal(messages, images, on_token);
        }

        // Text-only generation path
        let prompt = self.build_chat_prompt(&messages)?;

        // Start timing
        let generation_start = Instant::now();
        let mut first_token_time: Option<Instant> = None;

        // Create context for inference
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(4096))
            .with_n_batch(512);

        let mut ctx = model
            .new_context(&self.backend, ctx_params)
            .map_err(|e| format!("Failed to create context: {}", e))?;

        // Tokenize the prompt
        let tokens = model
            .str_to_token(&prompt, AddBos::Always)
            .map_err(|e| format!("Failed to tokenize: {}", e))?;

        let prompt_tokens = tokens.len() as u32;

        // Create batch and add tokens
        let mut batch = LlamaBatch::new(512, 1);
        let last_idx = tokens.len() - 1;
        for (i, token) in tokens.iter().enumerate() {
            batch
                .add(*token, i as i32, &[0], i == last_idx)
                .map_err(|e| format!("Failed to add token to batch: {}", e))?;
        }

        // Evaluate initial prompt
        ctx.decode(&mut batch)
            .map_err(|e| format!("Failed to decode prompt: {}", e))?;

        // Set up sampler using configurable parameters
        let params = &self.sampler_params;
        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::temp(params.temperature),
            LlamaSampler::top_p(params.top_p, params.top_k.max(1) as usize),
            LlamaSampler::dist(params.seed),
        ]);

        // Generate tokens
        let mut full_response = String::new();
        let mut n_cur = tokens.len();
        let mut tokens_generated: u32 = 0;

        loop {
            // Sample next token
            let new_token = sampler.sample(&ctx, batch.n_tokens() - 1);

            // Check for end of generation
            if model.is_eog_token(new_token) {
                break;
            }

            // Record first token time
            if first_token_time.is_none() {
                first_token_time = Some(Instant::now());
            }
            tokens_generated += 1;

            // Decode token to text
            let token_str = model
                .token_to_str(new_token, Special::Tokenize)
                .map_err(|e| format!("Failed to decode token: {}", e))?;

            // Stream the token; callback returns false to cancel
            if !on_token(&token_str) {
                break;
            }
            full_response.push_str(&token_str);

            // Prepare next batch
            batch.clear();
            batch
                .add(new_token, n_cur as i32, &[0], true)
                .map_err(|e| format!("Failed to add generated token: {}", e))?;

            n_cur += 1;

            // Decode
            ctx.decode(&mut batch)
                .map_err(|e| format!("Failed to decode: {}", e))?;
        }

        // Calculate and store metrics
        let total_generation_time_ms = generation_start.elapsed().as_secs_f64() * 1000.0;
        let time_to_first_token_ms = first_token_time
            .map(|t| (t - generation_start).as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        let tokens_per_second = if total_generation_time_ms > 0.0 {
            (tokens_generated as f64) / (total_generation_time_ms / 1000.0)
        } else {
            0.0
        };

        self.last_metrics = Some(GenerationMetrics {
            tokens_generated,
            prompt_tokens,
            time_to_first_token_ms,
            total_generation_time_ms,
            tokens_per_second,
        });

        llm_log!("Generated {} tokens in {:.1}ms ({:.1} tok/s)",
            tokens_generated, total_generation_time_ms, tokens_per_second);
        Ok(full_response)
    }

    /// Generate response using multimodal (vision) pipeline
    fn generate_multimodal<F>(
        &mut self,
        messages: Vec<ChatMessage>,
        images: Vec<Vec<u8>>,
        mut on_token: F,
    ) -> Result<String, String>
    where
        F: FnMut(&str) -> bool,
    {
        let model = self.model.as_ref().ok_or("No model loaded")?;
        let mtmd_ctx = self.mtmd_ctx.as_ref().ok_or("No multimodal context loaded")?;

        llm_log!("Multimodal generation with {} images", images.len());

        // Start timing
        let generation_start = Instant::now();
        let mut first_token_time: Option<Instant> = None;

        // Build text prompt from messages
        let prompt = self.build_chat_prompt(&messages)?;

        // Create context for inference with larger context for images
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(8192))
            .with_n_batch(512);

        let mut ctx = model
            .new_context(&self.backend, ctx_params)
            .map_err(|e| format!("Failed to create context: {}", e))?;

        // Convert images to MtmdBitmaps using the mtmd context
        let mut bitmaps: Vec<MtmdBitmap> = Vec::new();
        for (i, img_data) in images.iter().enumerate() {
            let bitmap = MtmdBitmap::from_buffer(mtmd_ctx, img_data)
                .map_err(|e| format!("Failed to decode image {}: {:?}", i, e))?;
            bitmaps.push(bitmap);
        }

        // Create input text for tokenization
        let input_text = MtmdInputText {
            text: prompt.clone(),
            add_special: true,
            parse_special: true,
        };

        // Create input chunks from prompt and images
        let bitmap_refs: Vec<&MtmdBitmap> = bitmaps.iter().collect();
        let input_chunks = mtmd_ctx
            .tokenize(input_text, &bitmap_refs)
            .map_err(|e| format!("Failed to tokenize multimodal input: {:?}", e))?;

        // Evaluate all chunks to populate the context
        let n_past = input_chunks
            .eval_chunks(mtmd_ctx, &ctx, 0, 0, 512, true)
            .map_err(|e| format!("Failed to evaluate multimodal input: {:?}", e))?;

        let prompt_tokens = n_past as u32;

        // Set up sampler using configurable parameters
        let params = &self.sampler_params;
        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::temp(params.temperature),
            LlamaSampler::top_p(params.top_p, params.top_k.max(1) as usize),
            LlamaSampler::dist(params.seed),
        ]);

        // Generate tokens
        let mut full_response = String::new();
        let mut n_cur = n_past as usize;
        let mut tokens_generated: u32 = 0;

        // Create batch for generation
        let mut batch = LlamaBatch::new(512, 1);

        loop {
            // For first token, we sample from the last position after eval
            // For subsequent tokens, we decode the batch first
            if batch.n_tokens() > 0 {
                ctx.decode(&mut batch)
                    .map_err(|e| format!("Failed to decode: {}", e))?;
            }

            // Sample next token
            let logit_idx = if batch.n_tokens() > 0 {
                batch.n_tokens() - 1
            } else {
                -1
            };

            let new_token = sampler.sample(&ctx, logit_idx);

            // Check for end of generation
            if model.is_eog_token(new_token) {
                break;
            }

            // Record first token time
            if first_token_time.is_none() {
                first_token_time = Some(Instant::now());
            }
            tokens_generated += 1;

            // Decode token to text
            let token_str = model
                .token_to_str(new_token, Special::Tokenize)
                .map_err(|e| format!("Failed to decode token: {}", e))?;

            // Stream the token; callback returns false to cancel
            if !on_token(&token_str) {
                break;
            }
            full_response.push_str(&token_str);

            // Prepare next batch
            batch.clear();
            batch
                .add(new_token, n_cur as i32, &[0], true)
                .map_err(|e| format!("Failed to add generated token: {}", e))?;

            n_cur += 1;
        }

        // Calculate and store metrics
        let total_generation_time_ms = generation_start.elapsed().as_secs_f64() * 1000.0;
        let time_to_first_token_ms = first_token_time
            .map(|t| (t - generation_start).as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        let tokens_per_second = if total_generation_time_ms > 0.0 {
            (tokens_generated as f64) / (total_generation_time_ms / 1000.0)
        } else {
            0.0
        };

        self.last_metrics = Some(GenerationMetrics {
            tokens_generated,
            prompt_tokens,
            time_to_first_token_ms,
            total_generation_time_ms,
            tokens_per_second,
        });

        llm_log!("Generated {} tokens in {:.1}ms ({:.1} tok/s)",
            tokens_generated, total_generation_time_ms, tokens_per_second);
        Ok(full_response)
    }

    /// Extract image data from multimodal messages
    fn extract_images(&self, messages: &[ChatMessage]) -> Vec<Vec<u8>> {
        let mut images = Vec::new();

        for msg in messages {
            if let ChatContent::Parts(parts) = &msg.content {
                for part in parts {
                    if let ChatContentPart::Image { image } = part {
                        if let Some(data) = decode_base64_image(image) {
                            images.push(data);
                        }
                    }
                }
            }
        }

        images
    }

    /// Extract text content from a ChatContent, inserting media markers for images
    fn extract_text_content_with_markers(content: &ChatContent) -> String {
        match content {
            ChatContent::Text(text) => text.clone(),
            ChatContent::Parts(parts) => {
                let marker = mtmd_default_marker();
                let mut image_markers = Vec::new();
                let mut text_parts = Vec::new();

                for part in parts {
                    match part {
                        ChatContentPart::Text { text } => text_parts.push(text.clone()),
                        ChatContentPart::Image { .. } => {
                            image_markers.push(marker.to_string());
                        }
                    }
                }

                // Put image markers at the beginning, then text
                let mut result = image_markers.join(" ");
                if !result.is_empty() && !text_parts.is_empty() {
                    result.push_str("\n");
                }
                result.push_str(&text_parts.join(" "));
                result
            }
        }
    }

    /// Build a chat prompt from messages using a simple template
    fn build_chat_prompt(&self, messages: &[ChatMessage]) -> Result<String, String> {
        let mut prompt = String::new();

        for msg in messages {
            let text_content = Self::extract_text_content_with_markers(&msg.content);

            match msg.role.as_str() {
                "system" => {
                    prompt.push_str("<|system|>\n");
                    prompt.push_str(&text_content);
                    prompt.push_str("\n");
                }
                "user" => {
                    prompt.push_str("<|user|>\n");
                    prompt.push_str(&text_content);
                    prompt.push_str("\n");
                }
                "assistant" => {
                    prompt.push_str("<|assistant|>\n");
                    prompt.push_str(&text_content);
                    prompt.push_str("\n");
                }
                _ => {
                    prompt.push_str("<|user|>\n");
                    prompt.push_str(&text_content);
                    prompt.push_str("\n");
                }
            }
        }

        // Add generation prompt
        prompt.push_str("<|assistant|>\n");

        Ok(prompt)
    }
}

impl Default for LlmEngine {
    fn default() -> Self {
        Self::new().expect("Failed to create LlmEngine")
    }
}

/// Global engine instance wrapped in Mutex for thread-safe access
pub static LLM_ENGINE: Mutex<Option<LlmEngine>> = Mutex::new(None);

// ── llama.cpp / ggml C-level log hook ───────────────────────────────────────
// `llama-cpp-2` routes internal C logs through `tracing`, which never reaches
// our JS Logger. We override both `llama_log_set` and `ggml_log_set` with our
// own callback that feeds directly into `_emit_log` (and thus the emitter).

unsafe extern "C" fn llama_log_hook(
    level: llama_cpp_sys_2::ggml_log_level,
    text: *const std::os::raw::c_char,
    _user_data: *mut std::os::raw::c_void,
) {
    if text.is_null() { return; }
    let msg = unsafe { std::ffi::CStr::from_ptr(text) }
        .to_string_lossy();
    let msg = msg.trim_end_matches('\n');
    if msg.is_empty() { return; }

    let lvl = match level {
        llama_cpp_sys_2::GGML_LOG_LEVEL_ERROR => "error",
        llama_cpp_sys_2::GGML_LOG_LEVEL_WARN  => "warn",
        _                                      => "info",
    };
    _emit_log(lvl, msg.to_string());
}

/// Install our log hook into llama.cpp and ggml so all C-level logs flow
/// through `_emit_log` and reach the JS Logger.
fn install_llama_log_hook() {
    unsafe {
        // Setting llama resets ggml too, so set ggml last to guarantee both.
        llama_cpp_sys_2::llama_log_set(
            Some(llama_log_hook),
            std::ptr::null_mut(),
        );
        llama_cpp_sys_2::ggml_log_set(
            Some(llama_log_hook),
            std::ptr::null_mut(),
        );
    }
}

/// Initialize the global LLM engine
pub fn init_engine() -> Result<(), String> {
    let mut guard = LLM_ENGINE.lock().map_err(|e| format!("Lock error: {}", e))?;

    if guard.is_none() {
        // Install C-level log hook before the backend touches llama.cpp at all.
        install_llama_log_hook();
        match LlmEngine::new() {
            Ok(engine) => {
                *guard = Some(engine);
            }
            Err(e) => {
                llm_log!(error, "Failed to create engine: {}", e);
                return Err(e);
            }
        }
    }
    Ok(())
}

/// Get a reference to the global engine (initializes if needed)
pub fn with_engine<F, R>(f: F) -> Result<R, String>
where
    F: FnOnce(&mut LlmEngine) -> Result<R, String>,
{
    init_engine()?;
    let mut guard = LLM_ENGINE.lock().map_err(|e| format!("Lock error: {}", e))?;
    let engine = guard.as_mut().ok_or("Engine not initialized")?;
    f(engine)
}

/// Extract model ID from a filename (removes .gguf extension)
pub fn model_id_from_filename(filename: &str) -> String {
    filename
        .trim_end_matches(".gguf")
        .trim_end_matches(".GGUF")
        .to_string()
}

/// Extract filename from a HuggingFace URL
pub fn filename_from_hf_url(url: &str) -> Option<String> {
    url.split('/').last().map(|s| s.to_string())
}

/// Find an mmproj file for a given model file
pub fn find_mmproj_for_model(model_path: &PathBuf) -> Option<PathBuf> {
    let parent = model_path.parent()?;
    let model_stem = model_path.file_stem()?.to_str()?;

    if let Ok(entries) = std::fs::read_dir(parent) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
                let lower_filename = filename.to_lowercase();
                if lower_filename.contains("mmproj") && lower_filename.ends_with(".gguf") {
                    // Prefer mmproj files that share a common prefix with the model
                    let model_prefix = model_stem.split('-').take(3).collect::<Vec<_>>().join("-");
                    if lower_filename.contains(&model_prefix.to_lowercase()) {
                        return Some(path);
                    }
                    return Some(path);
                }
            }
        }
    }

    None
}

/// Find an mmproj file by filename in a directory
pub fn find_mmproj_by_filename(models_dir: &PathBuf, mmproj_filename: &str) -> Option<PathBuf> {
    let path = models_dir.join(mmproj_filename);
    if path.exists() {
        Some(path)
    } else {
        None
    }
}

/// Decode a base64 image string to raw bytes
fn decode_base64_image(image_str: &str) -> Option<Vec<u8>> {
    use base64::Engine;

    // Strip data URL prefix if present
    let base64_data = if image_str.starts_with("data:") {
        image_str.split(',').nth(1)?
    } else {
        image_str
    };

    base64::prelude::BASE64_STANDARD.decode(base64_data.trim()).ok()
}

/// Check if a model file has an associated mmproj file
pub fn model_has_mmproj(model_path: &PathBuf) -> bool {
    find_mmproj_for_model(model_path).is_some()
}

// ============================================================================
// Tests
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_png_bytes() -> Vec<u8> {
        use base64::Engine;
        base64::prelude::BASE64_STANDARD
            .decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg==")
            .unwrap()
    }

    fn tiny_png_data_url() -> String {
        use base64::Engine;
        format!(
            "data:image/png;base64,{}",
            base64::prelude::BASE64_STANDARD.encode(tiny_png_bytes())
        )
    }

    #[test]
    fn test_decode_base64_raw() {
        use base64::Engine;
        let original = b"hello world";
        let encoded = base64::prelude::BASE64_STANDARD.encode(original);
        let decoded = decode_base64_image(&encoded).expect("raw base64 should decode");
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_decode_base64_data_url() {
        use base64::Engine;
        let original = b"image bytes here";
        let encoded = base64::prelude::BASE64_STANDARD.encode(original);
        let data_url = format!("data:image/png;base64,{}", encoded);
        let decoded = decode_base64_image(&data_url).expect("data URL should decode");
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_decode_base64_invalid_returns_none() {
        let result = decode_base64_image("not!!!valid_base64");
        assert!(result.is_none(), "Invalid base64 should return None");
    }

    #[test]
    fn test_decode_base64_png_produces_png_magic() {
        let decoded = decode_base64_image(&tiny_png_data_url()).expect("should decode");
        assert_eq!(&decoded[0..4], &[0x89, 0x50, 0x4E, 0x47], "Should be PNG magic bytes");
    }

    #[test]
    fn test_text_only_content_unchanged() {
        let content = ChatContent::Text("hello world".to_string());
        let result = LlmEngine::extract_text_content_with_markers(&content);
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_parts_text_only() {
        let content = ChatContent::Parts(vec![
            ChatContentPart::Text { text: "first".to_string() },
            ChatContentPart::Text { text: "second".to_string() },
        ]);
        let result = LlmEngine::extract_text_content_with_markers(&content);
        assert!(result.contains("first"));
        assert!(result.contains("second"));
    }

    #[test]
    fn test_parts_with_image_inserts_marker() {
        let marker = mtmd_default_marker().to_string();

        let content = ChatContent::Parts(vec![
            ChatContentPart::Image { image: "fake_base64_data".to_string() },
            ChatContentPart::Text { text: "what do you see?".to_string() },
        ]);
        let result = LlmEngine::extract_text_content_with_markers(&content);

        assert!(result.contains(&marker), "Should contain image marker");
        assert!(result.contains("what do you see?"), "Should contain text");
    }

    #[test]
    fn test_find_mmproj_in_same_dir() {
        use std::fs;
        let tmp = std::env::temp_dir().join("llm_plugin_test_mmproj");
        fs::create_dir_all(&tmp).unwrap();

        let model_path = tmp.join("gemma-4-E2B-it.gguf");
        let mmproj_path = tmp.join("gemma-4-E2B-it-mmproj.gguf");
        fs::write(&model_path, b"fake model").unwrap();
        fs::write(&mmproj_path, b"fake mmproj").unwrap();

        let found = find_mmproj_for_model(&model_path);
        assert!(found.is_some(), "Should find mmproj in same directory");
        assert_eq!(found.unwrap(), mmproj_path);

        fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn test_find_mmproj_not_found() {
        use std::fs;
        let tmp = std::env::temp_dir().join("llm_plugin_test_no_mmproj");
        fs::create_dir_all(&tmp).unwrap();

        let model_path = tmp.join("text-only-model.gguf");
        fs::write(&model_path, b"fake model").unwrap();

        let found = find_mmproj_for_model(&model_path);
        assert!(found.is_none(), "Should return None when no mmproj exists");

        fs::remove_dir_all(&tmp).unwrap();
    }
}
