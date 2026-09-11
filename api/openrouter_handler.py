#!/usr/bin/env python3
import os
import json
import logging
import time
import httpx # Use httpx for async requests
from fastapi.responses import StreamingResponse

# Import base class and custom exceptions
from api_handlers import BaseAPIHandler, ConfigError, BackendAPIError, HandlerError, get_http_client

logger = logging.getLogger("openrouter_handler")

class OpenRouterAPIHandler(BaseAPIHandler):
    """
    Asynchronous handler for OpenRouter API requests using httpx,
    with mapping for user-friendly model names.
    """
    OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self):
        super().__init__("openrouter")

        # {"name": "google/gemini-2.0-flash-exp:free", "parameters": "unknown"},
        # {"name": "google/gemma-3-27b-it:free", "parameters": "27B"},
        # {"name": "deepseek/deepseek-r1-zero:free", "parameters": "671B"},
        # {"name": "deepseek/deepseek-v3-base:free", "parameters": "671B"}, # Example model ID

        # --- Model Mapping ---
        # Dictionary mapping display names to actual model IDs and parameter sizes
        self.model_map = {
            # --- Add your desired mappings here ---
            # "gemma-3-27b-or": {
            #     "model_id": "google/gemma-3-27b-it:free",
            #     "parameters": "27B",
            #     "multimodal": True
            # },
            "Qwen3-VL-8B": {
                "model_id": "qwen/qwen3-vl-8b-instruct", 
                "parameters": "8B",
                "multimodal": True,
                "pro": False,
            },

            "OpenAI GPT-5": {
                "model_id": "openai/gpt-5-mini", 
                "parameters": "N/A",
                "multimodal": True,
                "pro": True,
            },

            "OpenAI GPT-4o-mini": {
                "model_id": "openai/gpt-4o-mini", 
                "parameters": "N/A",
                "multimodal": True,
                "pro": True,
            },
            "xAI Grok 4 Fast": {
                "model_id": "x-ai/grok-4-fast", 
                "parameters": "N/A",
                "multimodal": True,
                "pro": True,
            },
            "Nemotron Nano V2": {
                "model_id": "nvidia/nemotron-nano-9b-v2:free",
                "parameters": "9B",
                "multimodal": False 
            },
            "Nemotron Nano V2 VL": {
                "model_id": "nvidia/nemotron-nano-12b-v2-vl:free",
                "parameters": "12B",
                "multimodal": True
            },
            # Paid-only fast lane for Gemma 4 26B A4B. Same weights as the free
            # gemma-4-26b-a4b-it on Gemini's direct AI Studio proxy, but routed
            # through OpenRouter pinned to DeepInfra's dedicated paid capacity —
            # sub-second latency instead of Google's free-tier serving lane,
            # which queues badly under load (see gemini_handler.py).
            "gemma-4-26b-a4b-fast": {
                "model_id": "google/gemma-4-26b-a4b-it",
                "parameters": "26BA4",
                "multimodal": True,
                "pro": True,
                # Prefer cheap+fast providers in order, but DON'T hard-pin: OpenRouter's
                # DeepInfra pool is shared across all their customers and gets rate-limited
                # on its own under load (confirmed live: 429 "engine_overloaded" 3/3 tries
                # while Cloudflare/Novita served fine). allow_fallbacks left True so a bad
                # provider degrades to the next one instead of hard-failing the request.
                "provider_routing": {"order": ["deepinfra", "cloudflare", "novita"], "allow_fallbacks": True},
            }
            # "Skip Model Call": {
            #     "model_id": "deepseek/deepseek-chat-v3.1:free", # Example
            #     "parameters": "0B",
            #     "multimodal": False
            # }
            # Add more models following this pattern
            # "your-pretty-name": { "model_id": "actual/openrouter-model-id:tag", "parameters": "..."}
        }

        # Define supported models for display using the pretty names from the map
        self.models = [
            {"name": display_name, "parameters": model_info.get("parameters", "N/A"),
             "multimodal": model_info.get("multimodal", False), 
             "pro": model_info.get("pro", False) }
            for display_name, model_info in self.model_map.items()
        ]
        # --- End Model Mapping ---


        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            logger.error("OPENROUTER_API_KEY environment variable not set. OpenRouter handler will fail.")

        # Log the DISPLAY names that will be shown to the user
        logger.info("OpenRouterAPIHandler registered display models: %s", [m["name"] for m in self.models])

        # Base headers
        self.base_headers = {
            "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
            "Content-Type": "application/json",
            "HTTP-Referer": os.environ.get("OPENROUTER_SITE_URL", "http://localhost"),
            "X-Title": os.environ.get("OPENROUTER_APP_TITLE", "ObserverAI-FastAPI"),
            "User-Agent": "ObserverAI-FastAPI-Client/1.0"
        }

    async def handle_request(self, request_data: dict):
        """
        Process a /v1/chat/completions request asynchronously via OpenRouter.
        Translates display model name to actual OpenRouter model ID.
        Returns either dict (non-streaming) or StreamingResponse (streaming).
        """
        if not self.api_key:
            raise ConfigError("OPENROUTER_API_KEY is not configured on the server.")

        # --- Get Display Name and Translate to Actual Model ID ---
        display_model_name = request_data.get("model")
        if not display_model_name:
            raise ValueError("Request data must include a 'model' field (using the display name).")

        # --- Special Case: NULL Model ---
        if display_model_name == "NULL":
            logger.info("NULL model requested, returning empty response without API call")

            if request_data.get("stream", False):
                # Return streaming response for NULL
                return StreamingResponse(
                    self._generate_null_stream(),
                    media_type="text/event-stream"
                )
            else:
                # Return non-streaming response for NULL
                return {
                    "id": f"chatcmpl-null-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": "NULL",
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": " "
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 1,
                        "total_tokens": 1
                    }
                }
        # --- End Special Case ---

        # Look up the display name in our map
        model_info = self.model_map.get(display_model_name)
        if not model_info:
            # If the display name isn't found, the model is unsupported by this mapping
            logger.warning(f"Received request for unmapped OpenRouter model display name: {display_model_name}")
            raise ValueError(f"Model display name '{display_model_name}' is not recognized or supported.")

        actual_model_id = model_info.get("model_id")
        if not actual_model_id:
             # Should not happen if map is defined correctly, but good practice to check
             logger.error(f"Internal configuration error: Missing 'model_id' for display name '{display_model_name}' in model_map.")
             raise ConfigError(f"Internal mapping error for model '{display_model_name}'.")
        # --- End Translation ---

        # --- Prepare API Call ---
        # Create a copy of the request data to modify
        payload = request_data.copy()
        # *** IMPORTANT: Set the 'model' in the payload to the ACTUAL OpenRouter ID ***
        payload["model"] = actual_model_id
        # Pin provider routing when the model map asks for it (e.g. the fast
        # Gemma lane pinned to DeepInfra) so OpenRouter never falls back to a
        # slower/pricier provider behind the scenes.
        if model_info.get("provider_routing"):
            payload["provider"] = model_info["provider_routing"]

        # Update headers (in case API key was missing during init)
        headers = self.base_headers.copy()
        headers["Authorization"] = f"Bearer {self.api_key}"

        logger.info(f"Calling OpenRouter API: display_model='{display_model_name}', actual_model='{actual_model_id}', streaming={payload.get('stream', False)}")

        # --- Check for streaming ---
        if payload.get("stream", False):
            return StreamingResponse(
                self._stream_openrouter_response(headers, payload, display_model_name, actual_model_id),
                media_type="text/event-stream"
            )

        # --- Make Non-Streaming API Call using httpx ---
        try:
            client = get_http_client()
            response = await client.post(self.OPENROUTER_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            response_data = response.json()

        # --- Error Handling (keep as before) ---
        except httpx.RequestError as exc:
            logger.error(f"OpenRouter API request failed (network/connection): {exc}")
            raise BackendAPIError(f"Could not connect to OpenRouter API: {exc}", status_code=503) from exc
        except httpx.HTTPStatusError as exc:
            error_body = exc.response.text
            status_code = exc.response.status_code
            logger.error(f"OpenRouter API returned error {status_code} for model {actual_model_id}: {error_body[:500]}")
            try:
                error_json = exc.response.json()
                message = error_json.get("error", {}).get("message", error_body)
            except json.JSONDecodeError:
                message = error_body
            raise BackendAPIError(f"OpenRouter API Error ({status_code}): {message}", status_code=status_code) from exc
        except Exception as exc:
            logger.exception(f"An unexpected error occurred during OpenRouter API call for model {actual_model_id}")
            raise HandlerError(f"Unexpected error processing OpenRouter request: {exc}") from exc

        # --- Log Conversation (using display name for consistency if desired) ---
        prompt_text = ""
        response_text = ""
        try:
            messages = request_data.get("messages", [])
            if messages and isinstance(messages[-1].get("content"), str):
                prompt_text = messages[-1].get("content", "")[:500]
            if 'choices' in response_data and response_data['choices']:
                 choice = response_data['choices'][0]
                 if 'message' in choice and 'content' in choice['message']:
                      response_text = choice['message']['content'][:500]
        except Exception as log_parse_err:
            logger.warning(f"Could not parse prompt/response for logging: {log_parse_err}")

        # --- Conversation logging now handled centrally in compute.py ---

        # --- Return Response ---
        # Modify the response payload to show the *display name* instead of the actual ID? Optional.
        # If you want the client to see the pretty name in the response:
        if "model" in response_data:
             response_data["model"] = display_model_name # Replace actual ID with display name

        logger.info(f"Successfully processed OpenRouter request for display model '{display_model_name}'.")
        return response_data

    async def _stream_openrouter_response(self, headers: dict, payload: dict, display_model_name: str, actual_model_id: str):
        """Stream SSE chunks from OpenRouter API."""
        try:
            client = get_http_client()
            async with client.stream("POST", self.OPENROUTER_API_URL, headers=headers, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        # Replace actual model ID with display name in streaming chunks
                        chunk_data = line[6:]  # Remove "data: " prefix
                        if chunk_data != "[DONE]":
                            try:
                                chunk_json = json.loads(chunk_data)
                                if "model" in chunk_json:
                                    chunk_json["model"] = display_model_name
                                yield f"data: {json.dumps(chunk_json)}\n\n"
                            except json.JSONDecodeError:
                                # If we can't parse, just forward as-is
                                yield f"data: {chunk_data}\n\n"
                        else:
                            yield f"data: {chunk_data}\n\n"
        except httpx.RequestError as exc:
            logger.error(f"OpenRouter streaming API request failed: {exc}")
            yield f"data: {json.dumps({'error': f'Connection error: {exc}'})}\n\n"
        except httpx.HTTPStatusError as exc:
            error_body = exc.response.text
            logger.error(f"OpenRouter streaming API error {exc.response.status_code}: {error_body[:500]}")
            yield f"data: {json.dumps({'error': f'API error ({exc.response.status_code}): {error_body}'})}\n\n"
        except Exception as exc:
            logger.exception(f"Unexpected error in OpenRouter streaming for model {actual_model_id}")
            yield f"data: {json.dumps({'error': f'Unexpected error: {exc}'})}\n\n"

    async def _generate_null_stream(self):
        """Generate a minimal streaming response for the NULL model."""
        chunk_id = f"chatcmpl-null-{int(time.time())}"
        created = int(time.time())

        # Send initial chunk with a single space
        yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': 'NULL', 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ' '}, 'finish_reason': None}]})}\n\n"

        # Send final chunk marking completion
        yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': 'NULL', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"

        # Send [DONE] marker
        yield "data: [DONE]\n\n"
