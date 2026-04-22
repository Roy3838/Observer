// llm_engine.rs - Native LLM inference engine using llama.cpp with Metal acceleration
// This module is iOS-only and provides streaming text generation from GGUF models.
// Supports multimodal (vision) models via mmproj (multimodal projector) files.

use std::path::PathBuf;
use std::sync::Mutex;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel, Special};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::mtmd::{MtmdContext, MtmdContextParams, MtmdBitmap, MtmdInputText, mtmd_default_marker};

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
        })
    }

    /// Load a GGUF model from the given path
    /// Optionally loads mmproj file if present alongside the model
    pub fn load_model(&mut self, model_path: PathBuf, model_id: String) -> Result<(), String> {
        eprintln!("[LlmEngine] Loading model from: {:?}", model_path);

        if !model_path.exists() {
            return Err(format!("Model file not found: {:?}", model_path));
        }

        // Unload any existing model
        self.unload();

        // Configure model parameters for iOS Metal acceleration
        let model_params = LlamaModelParams::default()
            .with_n_gpu_layers(99); // Use Metal for all layers

        let model = LlamaModel::load_from_file(&self.backend, &model_path, &model_params)
            .map_err(|e| format!("Failed to load model: {}", e))?;

        eprintln!("[LlmEngine] Model loaded successfully");
        self.model = Some(model);
        self.model_path = Some(model_path.clone());
        self.loaded_model_id = Some(model_id.clone());

        // Try to find and load associated mmproj file
        let mmproj_path = find_mmproj_for_model(&model_path);
        if let Some(mmproj) = mmproj_path {
            eprintln!("[LlmEngine] Found mmproj file: {:?}", mmproj);
            if let Err(e) = self.load_mmproj(mmproj.clone()) {
                eprintln!("[LlmEngine] Warning: Failed to load mmproj: {}", e);
                // Continue without multimodal - not a fatal error
            }
        } else {
            eprintln!("[LlmEngine] No mmproj file found, model is text-only");
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

    /// Generate a response from chat messages with streaming callback
    /// Supports both text-only and multimodal (image + text) messages
    pub fn generate<F>(
        &self,
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

        eprintln!("[LlmEngine] Tokenized to {} tokens", tokens.len());

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

        // Set up sampler for generation
        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::temp(0.7),
            LlamaSampler::top_p(0.9, 1),
            LlamaSampler::dist(42),
        ]);

        // Generate tokens
        let mut full_response = String::new();
        let mut n_cur = tokens.len();

        for _ in 0..max_tokens {
            // Sample next token
            let new_token = sampler.sample(&ctx, batch.n_tokens() - 1);

            // Check for end of generation
            if model.is_eog_token(new_token) {
                eprintln!("[LlmEngine] End of generation");
                break;
            }

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

        eprintln!("[LlmEngine] Generated {} tokens", full_response.len());
        Ok(full_response)
    }

    /// Generate response using multimodal (vision) pipeline
    fn generate_multimodal<F>(
        &self,
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

        eprintln!("[LlmEngine] Evaluated {} tokens/embeddings", n_past);

        // Set up sampler for generation
        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::temp(0.7),
            LlamaSampler::top_p(0.9, 1),
            LlamaSampler::dist(42),
        ]);

        // Generate tokens
        let mut full_response = String::new();
        let mut n_cur = n_past as usize;

        // Create batch for generation
        let mut batch = LlamaBatch::new(512, 1);

        for _ in 0..max_tokens {
            // For first token, we sample from the last position after eval
            // For subsequent tokens, we decode the batch first
            if batch.n_tokens() > 0 {
                ctx.decode(&mut batch)
                    .map_err(|e| format!("Failed to decode: {}", e))?;
            }

            // Sample next token from last position
            let logit_idx = if batch.n_tokens() > 0 {
                batch.n_tokens() - 1
            } else {
                // First token after multimodal eval
                0
            };

            let new_token = sampler.sample(&ctx, logit_idx);

            // Check for end of generation
            if model.is_eog_token(new_token) {
                eprintln!("[LlmEngine] End of generation");
                break;
            }

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

        eprintln!("[LlmEngine] Generated {} chars", full_response.len());
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
