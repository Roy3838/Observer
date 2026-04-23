// llm_engine.rs - Native LLM inference engine using llama.cpp with Metal acceleration
// This module is iOS-only and provides streaming text generation from GGUF models.
// Supports multimodal (vision) models via mmproj (multimodal projector) files.

use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

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
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            seed: 42,
            repeat_penalty: 1.1,
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
}

impl LlmEngine {
    /// Create a new uninitialized engine
    pub fn new() -> Result<Self, String> {
        eprintln!("[LlmEngine] ========== BACKEND INIT START ==========");
        eprintln!("[LlmEngine] Calling LlamaBackend::init()...");

        let backend = match LlamaBackend::init() {
            Ok(b) => {
                eprintln!("[LlmEngine] ✅ LlamaBackend initialized successfully!");
                b
            }
            Err(e) => {
                eprintln!("[LlmEngine] ❌ LlamaBackend::init() FAILED: {:?}", e);
                eprintln!("[LlmEngine] Error type: {}", std::any::type_name_of_val(&e));
                eprintln!("[LlmEngine] ========== BACKEND INIT END ==========");
                return Err(format!("Failed to init llama backend: {}", e));
            }
        };

        eprintln!("[LlmEngine] ========== BACKEND INIT END ==========");
        Ok(Self {
            backend,
            model: None,
            model_path: None,
            loaded_model_id: None,
            mtmd_ctx: None,
            mmproj_path: None,
            sampler_params: SamplerParams::default(),
            last_metrics: None,
        })
    }

    /// Load a GGUF model from the given path.
    /// If mmproj_path is provided it is used directly; otherwise the models directory
    /// is scanned for any file containing "mmproj" in its name (auto-detect fallback).
    pub fn load_model(&mut self, model_path: PathBuf, model_id: String, mmproj_path: Option<PathBuf>) -> Result<(), String> {
        eprintln!("[LlmEngine] Loading model from: {:?}", model_path);
        eprintln!("[LlmEngine] mmproj: {:?}", mmproj_path);

        if !model_path.exists() {
            return Err(format!("Model file not found: {:?}", model_path));
        }

        self.unload();

        let model_params = LlamaModelParams::default()
            .with_n_gpu_layers(99);

        let model = LlamaModel::load_from_file(&self.backend, &model_path, &model_params)
            .map_err(|e| format!("Failed to load model: {}", e))?;

        eprintln!("[LlmEngine] Model loaded successfully");
        self.model = Some(model);
        self.model_path = Some(model_path.clone());
        self.loaded_model_id = Some(model_id.clone());

        // Resolve mmproj: explicit > auto-detect
        let resolved_mmproj = mmproj_path.or_else(|| find_mmproj_for_model(&model_path));
        if let Some(mmproj) = resolved_mmproj {
            eprintln!("[LlmEngine] Loading mmproj: {:?}", mmproj);
            if let Err(e) = self.load_mmproj(mmproj) {
                eprintln!("[LlmEngine] Warning: Failed to load mmproj: {}", e);
            }
        } else {
            eprintln!("[LlmEngine] No mmproj — model is text-only");
        }

        Ok(())
    }

    /// Load a multimodal projector (mmproj) file for vision support
    pub fn load_mmproj(&mut self, mmproj_path: PathBuf) -> Result<(), String> {
        eprintln!("[LlmEngine] Loading mmproj from: {:?}", mmproj_path);

        if !mmproj_path.exists() {
            return Err(format!("mmproj file not found: {:?}", mmproj_path));
        }

        let model = self.model.as_ref().ok_or("No model loaded")?;

        // Create mtmd context params with GPU acceleration (Metal)
        let mut mtmd_params = MtmdContextParams::default();
        mtmd_params.use_gpu = true; // Use Metal for projector

        let mmproj_str = mmproj_path.to_str()
            .ok_or_else(|| "Invalid mmproj path".to_string())?;

        let mtmd_ctx = MtmdContext::init_from_file(mmproj_str, model, &mtmd_params)
            .map_err(|e| format!("Failed to create mtmd context: {:?}", e))?;

        eprintln!("[LlmEngine] mmproj loaded successfully, multimodal enabled");
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
            eprintln!("[LlmEngine] Unloading model");
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
        eprintln!("[LlmEngine] Setting sampler params: temp={}, top_p={}, top_k={}, seed={}, repeat_penalty={}",
            params.temperature, params.top_p, params.top_k, params.seed, params.repeat_penalty);
        self.sampler_params = params;
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
        max_tokens: u32,
        mut on_token: F,
    ) -> Result<String, String>
    where
        F: FnMut(&str),
    {
        let model = self.model.as_ref().ok_or("No model loaded")?;

        // Extract images from messages for multimodal processing
        let images = self.extract_images(&messages);
        let has_images = !images.is_empty();

        if has_images && self.mtmd_ctx.is_none() {
            eprintln!("[LlmEngine] Warning: Images provided but no mmproj loaded, ignoring images");
        }

        // If we have images and multimodal context, use mtmd pipeline
        if has_images && self.mtmd_ctx.is_some() {
            return self.generate_multimodal(messages, images, max_tokens, on_token);
        }

        // Text-only generation path
        let prompt = self.build_chat_prompt(&messages)?;
        eprintln!("[LlmEngine] Prompt length: {} chars", prompt.len());

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
        eprintln!("[LlmEngine] Tokenized to {} tokens", prompt_tokens);

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

        for _ in 0..max_tokens {
            // Sample next token
            let new_token = sampler.sample(&ctx, batch.n_tokens() - 1);

            // Check for end of generation
            if model.is_eog_token(new_token) {
                eprintln!("[LlmEngine] End of generation");
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

            // Stream the token
            on_token(&token_str);
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

        eprintln!("[LlmEngine] Generated {} tokens in {:.1}ms ({:.1} tok/s)",
            tokens_generated, total_generation_time_ms, tokens_per_second);
        Ok(full_response)
    }

    /// Generate response using multimodal (vision) pipeline
    fn generate_multimodal<F>(
        &mut self,
        messages: Vec<ChatMessage>,
        images: Vec<Vec<u8>>,
        max_tokens: u32,
        mut on_token: F,
    ) -> Result<String, String>
    where
        F: FnMut(&str),
    {
        let model = self.model.as_ref().ok_or("No model loaded")?;
        let mtmd_ctx = self.mtmd_ctx.as_ref().ok_or("No multimodal context loaded")?;

        eprintln!("[LlmEngine] Starting multimodal generation with {} images", images.len());

        // Start timing
        let generation_start = Instant::now();
        let mut first_token_time: Option<Instant> = None;

        // Build text prompt from messages
        let prompt = self.build_chat_prompt(&messages)?;
        eprintln!("[LlmEngine] Multimodal prompt length: {} chars", prompt.len());

        // Create context for inference with larger context for images
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(8192)) // Larger context for image tokens
            .with_n_batch(512);

        let mut ctx = model
            .new_context(&self.backend, ctx_params)
            .map_err(|e| format!("Failed to create context: {}", e))?;

        // Convert images to MtmdBitmaps using the mtmd context
        let mut bitmaps: Vec<MtmdBitmap> = Vec::new();
        for (i, img_data) in images.iter().enumerate() {
            eprintln!("[LlmEngine] Processing image {} ({} bytes)", i, img_data.len());
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
        // The mtmd context tokenizes the prompt and embeds images at appropriate positions
        let bitmap_refs: Vec<&MtmdBitmap> = bitmaps.iter().collect();
        let input_chunks = mtmd_ctx
            .tokenize(input_text, &bitmap_refs)
            .map_err(|e| format!("Failed to tokenize multimodal input: {:?}", e))?;

        eprintln!("[LlmEngine] Created {} input chunks", input_chunks.len());

        // Evaluate all chunks to populate the context
        let n_past = input_chunks
            .eval_chunks(mtmd_ctx, &ctx, 0, 0, 512, true)
            .map_err(|e| format!("Failed to evaluate multimodal input: {:?}", e))?;

        let prompt_tokens = n_past as u32;
        eprintln!("[LlmEngine] Evaluated {} tokens/embeddings", n_past);

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

        for _ in 0..max_tokens {
            // For first token, we sample from the last position after eval
            // For subsequent tokens, we decode the batch first
            if batch.n_tokens() > 0 {
                ctx.decode(&mut batch)
                    .map_err(|e| format!("Failed to decode: {}", e))?;
            }

            // Sample next token.
            // After eval_chunks (first iteration, empty batch): use -1 which means
            // "last position with logits" — eval_chunks sets logits only for the final token.
            // After our own decode (subsequent iterations): use the last token in the batch.
            let logit_idx = if batch.n_tokens() > 0 {
                batch.n_tokens() - 1
            } else {
                -1
            };

            let new_token = sampler.sample(&ctx, logit_idx);

            // Check for end of generation
            if model.is_eog_token(new_token) {
                eprintln!("[LlmEngine] End of generation");
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

            // Stream the token
            on_token(&token_str);
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

        eprintln!("[LlmEngine] Generated {} tokens in {:.1}ms ({:.1} tok/s)",
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
    /// The media marker tells the mtmd tokenizer where to embed image data
    /// Images are placed at the beginning of the content for better model attention
    fn extract_text_content_with_markers(content: &ChatContent) -> String {
        match content {
            ChatContent::Text(text) => text.clone(),
            ChatContent::Parts(parts) => {
                let marker = mtmd_default_marker();
                let mut image_markers = Vec::new();
                let mut text_parts = Vec::new();

                // Separate images and text, collect image markers first
                for part in parts {
                    match part {
                        ChatContentPart::Text { text } => text_parts.push(text.clone()),
                        ChatContentPart::Image { .. } => {
                            // Collect media marker for each image
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
    /// For multimodal messages, inserts media markers where images should be embedded
    fn build_chat_prompt(&self, messages: &[ChatMessage]) -> Result<String, String> {
        // Use a generic chat template that works with most models
        let mut prompt = String::new();

        for msg in messages {
            // Use markers version to include image placeholders
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

/// Initialize the global LLM engine
pub fn init_engine() -> Result<(), String> {
    eprintln!("[LlmEngine] init_engine() called");
    let mut guard = LLM_ENGINE.lock().map_err(|e| {
        eprintln!("[LlmEngine] ❌ Failed to acquire lock: {}", e);
        format!("Lock error: {}", e)
    })?;

    if guard.is_none() {
        eprintln!("[LlmEngine] Engine not yet initialized, creating new...");
        match LlmEngine::new() {
            Ok(engine) => {
                *guard = Some(engine);
                eprintln!("[LlmEngine] ✅ Global engine initialized successfully");
            }
            Err(e) => {
                eprintln!("[LlmEngine] ❌ Failed to create engine: {}", e);
                return Err(e);
            }
        }
    } else {
        eprintln!("[LlmEngine] Engine already initialized");
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
    // HF URLs look like: https://huggingface.co/user/repo/resolve/main/model.gguf
    url.split('/').last().map(|s| s.to_string())
}

/// Find an mmproj file for a given model file
/// Looks for files with "mmproj" in the name in the same directory
pub fn find_mmproj_for_model(model_path: &PathBuf) -> Option<PathBuf> {
    let parent = model_path.parent()?;
    let model_stem = model_path.file_stem()?.to_str()?;

    // Look for mmproj files in the same directory
    if let Ok(entries) = std::fs::read_dir(parent) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
                // Check if this is an mmproj file
                let lower_filename = filename.to_lowercase();
                if lower_filename.contains("mmproj") && lower_filename.ends_with(".gguf") {
                    // Prefer mmproj files that share a common prefix with the model
                    // e.g., gemma-4-E2B-it.gguf -> gemma-4-E2B-it-mmproj.gguf
                    let model_prefix = model_stem.split('-').take(3).collect::<Vec<_>>().join("-");
                    if lower_filename.contains(&model_prefix.to_lowercase()) {
                        eprintln!("[LlmEngine] Found matching mmproj: {}", filename);
                        return Some(path);
                    }
                    // Otherwise, return any mmproj file found
                    eprintln!("[LlmEngine] Found mmproj file: {}", filename);
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
/// Handles both raw base64 and data URLs (data:image/...;base64,...)
fn decode_base64_image(image_str: &str) -> Option<Vec<u8>> {
    use base64::Engine;

    // Strip data URL prefix if present
    let base64_data = if image_str.starts_with("data:") {
        // Format: data:image/png;base64,<base64data>
        image_str.split(',').nth(1)?
    } else {
        image_str
    };

    // Decode base64
    base64::prelude::BASE64_STANDARD.decode(base64_data.trim()).ok()
}

/// Check if a model file has an associated mmproj file
pub fn model_has_mmproj(model_path: &PathBuf) -> bool {
    find_mmproj_for_model(model_path).is_some()
}

// ============================================================================
// Tests - run with: cd app/mobile && cargo test -- --nocapture
// For integration tests with real model files:
//   GGUF_MODEL_PATH=/path/to/model.gguf MMPROJ_PATH=/path/to/mmproj.gguf cargo test -- --nocapture --include-ignored
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    // Minimal valid 1x1 PNG — fine for unit tests (magic bytes / base64 round-trips).
    // NOT suitable for mtmd which requires ≥2x2; use test_image_data_url() in integration tests.
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

    // ===== decode_base64_image =====

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
    fn test_decode_base64_jpeg_data_url() {
        use base64::Engine;
        let original = b"\xFF\xD8\xFF\xE0 fake jpeg";
        let encoded = base64::prelude::BASE64_STANDARD.encode(original);
        let data_url = format!("data:image/jpeg;base64,{}", encoded);
        let decoded = decode_base64_image(&data_url).expect("jpeg data URL should decode");
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

    // ===== extract_text_content_with_markers =====

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
        eprintln!("[test] text-only parts result: {:?}", result);
        assert!(result.contains("first"));
        assert!(result.contains("second"));
        // No image markers
        let marker = mtmd_default_marker().to_string();
        assert!(!result.contains(&marker), "Should not contain image marker for text-only content");
    }

    #[test]
    fn test_parts_with_image_inserts_marker() {
        let marker = mtmd_default_marker().to_string();
        eprintln!("[test] mtmd marker string: {:?}", marker);

        let content = ChatContent::Parts(vec![
            ChatContentPart::Image { image: "fake_base64_data".to_string() },
            ChatContentPart::Text { text: "what do you see?".to_string() },
        ]);
        let result = LlmEngine::extract_text_content_with_markers(&content);
        eprintln!("[test] single-image result: {:?}", result);

        assert!(result.contains(&marker), "Should contain image marker, got: {:?}", result);
        assert!(result.contains("what do you see?"), "Should contain text");
        let marker_pos = result.find(&marker).unwrap();
        let text_pos = result.find("what do you see?").unwrap();
        assert!(marker_pos < text_pos, "Image marker must come BEFORE text");
    }

    #[test]
    fn test_parts_multiple_images_get_multiple_markers() {
        let marker = mtmd_default_marker().to_string();
        let content = ChatContent::Parts(vec![
            ChatContentPart::Image { image: "img1".to_string() },
            ChatContentPart::Image { image: "img2".to_string() },
            ChatContentPart::Text { text: "compare them".to_string() },
        ]);
        let result = LlmEngine::extract_text_content_with_markers(&content);
        eprintln!("[test] two-image result: {:?}", result);

        let count = result.matches(&marker).count();
        assert_eq!(count, 2, "Two images should produce exactly 2 markers, got {} in: {:?}", count, result);
    }

    // ===== extract_images and build_chat_prompt =====
    // These methods need &self but don't use the backend/model.
    // We use with_engine() so the backend is initialized exactly once across all tests.

    #[test]
    fn test_extract_images_from_text_message_is_empty() {
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: ChatContent::Text("no images here".to_string()),
        }];
        with_engine(|engine| {
            let images = engine.extract_images(&messages);
            eprintln!("[test] images from text-only: {}", images.len());
            assert!(images.is_empty());
            Ok(())
        }).expect("Engine failed");
    }

    #[test]
    fn test_extract_images_finds_base64_image() {
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: ChatContent::Parts(vec![
                ChatContentPart::Image { image: tiny_png_data_url() },
                ChatContentPart::Text { text: "what is this?".to_string() },
            ]),
        }];
        with_engine(|engine| {
            let images = engine.extract_images(&messages);
            eprintln!("[test] extracted {} image(s), first is {} bytes", images.len(), images.first().map(|b| b.len()).unwrap_or(0));
            assert_eq!(images.len(), 1, "Should extract exactly 1 image");
            assert!(!images[0].is_empty());
            assert_eq!(&images[0][0..4], &[0x89, 0x50, 0x4E, 0x47], "Should be PNG magic");
            Ok(())
        }).expect("Engine failed");
    }

    #[test]
    fn test_extract_images_multiple_messages() {
        let messages = vec![
            ChatMessage {
                role: "user".to_string(),
                content: ChatContent::Parts(vec![
                    ChatContentPart::Image { image: tiny_png_data_url() },
                    ChatContentPart::Text { text: "first image".to_string() },
                ]),
            },
            ChatMessage {
                role: "user".to_string(),
                content: ChatContent::Parts(vec![
                    ChatContentPart::Image { image: tiny_png_data_url() },
                    ChatContentPart::Text { text: "second image".to_string() },
                ]),
            },
        ];
        with_engine(|engine| {
            let images = engine.extract_images(&messages);
            eprintln!("[test] extracted {} image(s) from 2 messages", images.len());
            assert_eq!(images.len(), 2, "Should extract 2 images from 2 messages");
            Ok(())
        }).expect("Engine failed");
    }

    // ===== build_chat_prompt =====

    #[test]
    fn test_build_chat_prompt_text_only() {
        let messages = vec![
            ChatMessage { role: "system".to_string(), content: ChatContent::Text("You are helpful.".to_string()) },
            ChatMessage { role: "user".to_string(), content: ChatContent::Text("What is 2+2?".to_string()) },
        ];
        with_engine(|engine| {
            let prompt = engine.build_chat_prompt(&messages)?;
            eprintln!("[test] built prompt:\n{}", prompt);
            assert!(prompt.contains("<|system|>"));
            assert!(prompt.contains("You are helpful."));
            assert!(prompt.contains("<|user|>"));
            assert!(prompt.contains("What is 2+2?"));
            assert!(prompt.ends_with("<|assistant|>\n"));
            Ok(())
        }).expect("Engine failed");
    }

    #[test]
    fn test_build_chat_prompt_with_image_contains_marker() {
        let marker = mtmd_default_marker().to_string();
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: ChatContent::Parts(vec![
                ChatContentPart::Image { image: tiny_png_data_url() },
                ChatContentPart::Text { text: "Describe this image.".to_string() },
            ]),
        }];
        with_engine(|engine| {
            let prompt = engine.build_chat_prompt(&messages)?;
            eprintln!("[test] multimodal prompt:\n{}", prompt);
            assert!(prompt.contains(&marker), "Prompt must contain <__media__> marker, got: {:?}", prompt);
            assert!(prompt.contains("Describe this image."));
            Ok(())
        }).expect("Engine failed");
    }

    // ===== find_mmproj_for_model =====

    #[test]
    fn test_find_mmproj_in_same_dir() {
        use std::fs;
        let tmp = std::env::temp_dir().join("observer_test_mmproj");
        fs::create_dir_all(&tmp).unwrap();

        // Create dummy model and mmproj files
        let model_path = tmp.join("gemma-4-E2B-it.gguf");
        let mmproj_path = tmp.join("gemma-4-E2B-it-mmproj.gguf");
        fs::write(&model_path, b"fake model").unwrap();
        fs::write(&mmproj_path, b"fake mmproj").unwrap();

        let found = find_mmproj_for_model(&model_path);
        eprintln!("[test] find_mmproj result: {:?}", found);
        assert!(found.is_some(), "Should find mmproj in same directory");
        assert_eq!(found.unwrap(), mmproj_path);

        fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn test_find_mmproj_not_found() {
        use std::fs;
        let tmp = std::env::temp_dir().join("observer_test_no_mmproj");
        fs::create_dir_all(&tmp).unwrap();

        let model_path = tmp.join("text-only-model.gguf");
        fs::write(&model_path, b"fake model").unwrap();

        let found = find_mmproj_for_model(&model_path);
        eprintln!("[test] find_mmproj (no mmproj) result: {:?}", found);
        assert!(found.is_none(), "Should return None when no mmproj exists");

        fs::remove_dir_all(&tmp).unwrap();
    }

    // ===== Integration tests (require actual model files) =====
    // Run with: GGUF_MODEL_PATH=... MMPROJ_PATH=... cargo test -- --nocapture --include-ignored

    #[test]
    #[ignore = "Requires GGUF_MODEL_PATH env var pointing to a real GGUF model file"]
    fn test_integration_load_model() {
        let model_path = PathBuf::from(
            std::env::var("GGUF_MODEL_PATH").expect("Set GGUF_MODEL_PATH to a .gguf file")
        );
        eprintln!("[integration] Loading model: {:?}", model_path);
        with_engine(|engine| {
            engine.load_model(model_path.clone(), "test-model".to_string(), None)?;
            eprintln!("[integration] is_loaded: {}", engine.is_loaded());
            eprintln!("[integration] is_multimodal: {}", engine.is_multimodal());
            assert!(engine.is_loaded());
            Ok(())
        }).expect("Engine failed");
    }

    #[test]
    #[ignore = "Requires GGUF_MODEL_PATH + MMPROJ_PATH env vars"]
    fn test_integration_multimodal_with_image() {
        let model_path = PathBuf::from(std::env::var("GGUF_MODEL_PATH").expect("Set GGUF_MODEL_PATH"));
        let mmproj_path = PathBuf::from(std::env::var("MMPROJ_PATH").expect("Set MMPROJ_PATH"));

        with_engine(|engine| {
            eprintln!("[integration] Loading model: {:?}", model_path);
            engine.load_model(model_path.clone(), "test-model".to_string(), Some(mmproj_path.clone()))?;
            eprintln!("[integration] is_multimodal: {}", engine.is_multimodal());
            assert!(engine.is_multimodal(), "Should be multimodal after loading mmproj");

            // Load a real image: TEST_IMAGE_PATH env var (any JPEG/PNG from your phone works)
            let image_data_url = {
                use base64::Engine;
                let image_path = std::env::var("TEST_IMAGE_PATH")
                    .expect("Set TEST_IMAGE_PATH to a real JPEG or PNG file (mtmd requires ≥2x2)");
                let bytes = std::fs::read(&image_path)
                    .unwrap_or_else(|e| panic!("Failed to read TEST_IMAGE_PATH {}: {}", image_path, e));
                let ext = std::path::Path::new(&image_path)
                    .extension().and_then(|e| e.to_str()).unwrap_or("jpeg");
                let mime = if ext == "png" { "image/png" } else { "image/jpeg" };
                eprintln!("[integration] Loaded test image: {} bytes ({})", bytes.len(), image_path);
                format!("data:{};base64,{}", mime, base64::prelude::BASE64_STANDARD.encode(&bytes))
            };

            let messages = vec![ChatMessage {
                role: "user".to_string(),
                content: ChatContent::Parts(vec![
                    ChatContentPart::Image { image: image_data_url },
                    ChatContentPart::Text { text: "What do you see in this image? Describe briefly.".to_string() },
                ]),
            }];

            let images = engine.extract_images(&messages);
            eprintln!("[integration] Extracted {} image(s), sizes: {:?}", images.len(), images.iter().map(|i| i.len()).collect::<Vec<_>>());
            assert_eq!(images.len(), 1);

            let prompt = engine.build_chat_prompt(&messages)?;
            eprintln!("[integration] Prompt:\n{}", prompt);

            eprintln!("[integration] Starting multimodal generation...");
            let mut token_count = 0usize;
            let result = engine.generate(messages, 100, |token| {
                eprint!("{}", token);
                token_count += 1;
            });
            eprintln!("\n[integration] token_count={}", token_count);
            match result {
                Ok(r) => eprintln!("[integration] Response: {:?}", r),
                Err(e) => return Err(format!("Generation failed: {}", e)),
            }
            Ok(())
        }).expect("Engine failed");
    }

    #[test]
    #[ignore = "Requires GGUF_MODEL_PATH env var; tests text-only generation path"]
    fn test_integration_text_only_generation() {
        let model_path = PathBuf::from(std::env::var("GGUF_MODEL_PATH").expect("Set GGUF_MODEL_PATH"));
        with_engine(|engine| {
            eprintln!("[integration] Loading model: {:?}", model_path);
            engine.load_model(model_path.clone(), "test-model".to_string(), None)?;

            let messages = vec![
                ChatMessage { role: "user".to_string(), content: ChatContent::Text("Say hello.".to_string()) },
            ];

            eprintln!("[integration] Starting text generation...");
            let mut token_count = 0usize;
            let result = engine.generate(messages, 50, |token| {
                eprint!("{}", token);
                token_count += 1;
            });
            eprintln!("\n[integration] tokens={}", token_count);
            match result {
                Ok(r) => eprintln!("[integration] Response: {:?}", r),
                Err(e) => return Err(format!("Text generation failed: {}", e)),
            }
            Ok(())
        }).expect("Engine failed");
    }
}
