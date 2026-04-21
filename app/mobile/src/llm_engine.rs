// llm_engine.rs - Native LLM inference engine using llama.cpp with Metal acceleration
// This module is iOS-only and provides streaming text generation from GGUF models.

use std::path::PathBuf;
use std::sync::Mutex;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel, Special};
use llama_cpp_2::sampling::LlamaSampler;

/// Metadata for a downloaded GGUF model
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NativeModelInfo {
    pub id: String,           // Unique identifier (filename without extension)
    pub name: String,         // Display name
    pub filename: String,     // Actual filename on disk
    pub size_bytes: u64,      // File size
    pub hf_url: Option<String>, // Original HuggingFace URL if known
}

/// Message format for chat completion
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// LLM Engine state - holds the loaded model and context
pub struct LlmEngine {
    backend: LlamaBackend,
    model: Option<LlamaModel>,
    model_path: Option<PathBuf>,
    loaded_model_id: Option<String>,
}

impl LlmEngine {
    /// Create a new uninitialized engine
    pub fn new() -> Result<Self, String> {
        let backend = LlamaBackend::init().map_err(|e| format!("Failed to init llama backend: {}", e))?;

        Ok(Self {
            backend,
            model: None,
            model_path: None,
            loaded_model_id: None,
        })
    }

    /// Load a GGUF model from the given path
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
        self.model_path = Some(model_path);
        self.loaded_model_id = Some(model_id);

        Ok(())
    }

    /// Unload the current model and free memory
    pub fn unload(&mut self) {
        if self.model.is_some() {
            eprintln!("[LlmEngine] Unloading model");
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

        // Build prompt from chat messages using chat template
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

    /// Build a chat prompt from messages using a simple template
    fn build_chat_prompt(&self, messages: &[ChatMessage]) -> Result<String, String> {
        // Use a generic chat template that works with most models
        let mut prompt = String::new();

        for msg in messages {
            match msg.role.as_str() {
                "system" => {
                    prompt.push_str("<|system|>\n");
                    prompt.push_str(&msg.content);
                    prompt.push_str("\n");
                }
                "user" => {
                    prompt.push_str("<|user|>\n");
                    prompt.push_str(&msg.content);
                    prompt.push_str("\n");
                }
                "assistant" => {
                    prompt.push_str("<|assistant|>\n");
                    prompt.push_str(&msg.content);
                    prompt.push_str("\n");
                }
                _ => {
                    prompt.push_str("<|user|>\n");
                    prompt.push_str(&msg.content);
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
    let mut guard = LLM_ENGINE.lock().map_err(|e| format!("Lock error: {}", e))?;
    if guard.is_none() {
        *guard = Some(LlmEngine::new()?);
        eprintln!("[LlmEngine] Global engine initialized");
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
