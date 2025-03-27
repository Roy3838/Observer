#!/usr/bin/env python3
import http.server
import socketserver
import urllib.request
import urllib.error
import ssl
import subprocess
import os
import sys
import argparse
import logging
import signal
import socket
import threading
import time
import json
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('ollama-proxy')

class OllamaProxy(http.server.BaseHTTPRequestHandler):
    # Quieter logs - only errors
    def log_message(self, format, *args):
        if args[1][0] in ['4', '5']:  # Only log 4xx and 5xx errors
            logger.error("%s - %s", self.address_string(), format % args)
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()
    
    def do_GET(self):
        self.proxy_request("GET")
    
    def do_POST(self):
        self.proxy_request("POST")
    
    def send_cors_headers(self):
        origin = self.headers.get('Origin', '')
        allowed_origins = ['http://localhost:3000', 'http://localhost:3001', 'https://localhost:3000']
        
        if origin in allowed_origins or self.server.dev_mode:
            self.send_header("Access-Control-Allow-Origin", origin or "*")
        else:
            self.send_header("Access-Control-Allow-Origin", "*")
            
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, User-Agent")
        self.send_header("Access-Control-Allow-Credentials", "true")
        self.send_header("Access-Control-Max-Age", "86400")  # 24 hours
    
    def translate_openai_to_ollama(self, body):
        """
        Translate OpenAI v1 chat completions format to Ollama native format when images are present
        """
        try:
            request_data = json.loads(body)
            
            # Check if this is a chat completion request with a message that has content
            if not (
                'messages' in request_data and 
                isinstance(request_data['messages'], list) and 
                len(request_data['messages']) > 0 and 
                'content' in request_data['messages'][0]
            ):
                return None, body  # Not a chat request or missing content, pass through
            
            # Extract model name and first message content
            model = request_data.get('model', '')
            first_message = request_data['messages'][0]
            content = first_message.get('content', '')
            
            # Check if this is a multimodal request (content is a list with images)
            has_images = False
            prompt_text = ""
            images = []
            
            if isinstance(content, list):
                # Process multimodal content
                for item in content:
                    if isinstance(item, dict):
                        if item.get('type') == 'text':
                            prompt_text += item.get('text', '')
                        elif item.get('type') == 'image_url':
                            # Extract image URL - may be base64 or actual URL
                            image_url = item.get('image_url', {}).get('url', '')
                            if image_url:
                                images.append(image_url)
                                has_images = True
            else:
                # Simple text content
                prompt_text = content
            
            # Return the original body if no images are found
            if not has_images:
                return None, body
            
            # Create the Ollama native format for requests with images
            ollama_request = {
                'model': model,
                'prompt': prompt_text,
                'images': images,
                'stream': request_data.get('stream', False)
            }
            
            # Add any other parameters that should be forwarded
            for key in ['temperature', 'top_p', 'top_k', 'seed']:
                if key in request_data:
                    ollama_request[key] = request_data[key]
            
            logger.info(f"Translating request to Ollama native format for {len(images)} images")
            
            # Return new path and body for Ollama's native API
            return "/api/generate", json.dumps(ollama_request).encode('utf-8')
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            return None, body
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return None, body
    
    def proxy_request(self, method):
        target_path = self.path
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length) if content_length > 0 else None
        
        # If this is a POST to chat completions endpoint, check if it needs translation
        if method == "POST" and body and self.path == '/v1/chat/completions':
            new_path, new_body = self.translate_openai_to_ollama(body)
            if new_path:
                target_path = new_path
                body = new_body
        
        target_url = f"http://localhost:11434{target_path}"
        
        req = urllib.request.Request(target_url, data=body, method=method)
        
        # Forward relevant headers
        headers_to_forward = ['Content-Type', 'Authorization', 'User-Agent']
        for header in headers_to_forward:
            if header in self.headers:
                req.add_header(header, self.headers[header])
        
        # Use a much longer timeout for /api/generate endpoint
        timeout = 300 if target_path == '/api/generate' else 60  # 5 minutes for generate
        
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                self.send_response(response.status)
                
                # Forward response headers excluding problematic ones
                for key, val in response.getheaders():
                    if key.lower() not in ['transfer-encoding', 'connection', 'content-length']:
                        self.send_header(key, val)
                
                self.send_cors_headers()
                self.end_headers()
                
                # For native API responses, we need to translate back to OpenAI format
                if target_path == '/api/generate' and self.path == '/v1/chat/completions':
                    # Read the entire response
                    response_data = response.read()
                    try:
                        # Parse the response from Ollama's native format
                        ollama_response = json.loads(response_data)
                        
                        # Convert to OpenAI-compatible format
                        openai_response = {
                            "id": f"chatcmpl-{time.time()}",
                            "object": "chat.completion",
                            "created": int(time.time()),
                            "model": ollama_response.get("model", "unknown"),
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": ollama_response.get("response", "")
                                    },
                                    "finish_reason": "stop"
                                }
                            ],
                            "usage": {
                                "prompt_tokens": -1,
                                "completion_tokens": -1,
                                "total_tokens": -1
                            }
                        }
                        
                        # Write the transformed response
                        transformed_response = json.dumps(openai_response).encode('utf-8')
                        self.wfile.write(transformed_response)
                    except json.JSONDecodeError:
                        # If we can't parse it, return as-is
                        self.wfile.write(response_data)
                else:
                    # Stream the response back for non-translated requests
                    while True:
                        chunk = response.read(4096)  # Read in chunks
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                    
        except urllib.error.HTTPError as e:
            # Forward HTTP errors from the target server
            self.send_response(e.code)
            self.send_cors_headers()
            self.end_headers()
            if e.fp:
                self.wfile.write(e.read())
            else:
                self.wfile.write(f"Error: {str(e)}".encode())
        
        except socket.timeout:
            logger.error(f"Request to {target_url} timed out")
            self.send_response(504)  # Gateway Timeout
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(f"Request to Ollama timed out. For large models, the first request may take longer.".encode())
                
        except Exception as e:
            logger.error(f"Proxy error: {str(e)}")
            self.send_response(502)  # Bad Gateway
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(f"Proxy error: {str(e)}".encode())

def check_ollama_running():
    """Check if Ollama is already running on port 11434"""
    try:
        with urllib.request.urlopen("http://localhost:11434/api/version", timeout=2) as response:
            if response.status == 200:
                version = response.read().decode('utf-8')
                logger.info(f"Ollama server is running: {version}")
                return True
    except Exception as e:
        logger.debug(f"Ollama server is not running: {str(e)}")
        return False

def start_ollama_server():
    """Start Ollama server as a subprocess and capture its logs"""
    try:
        logger.info("Starting Ollama server...")
        process = subprocess.Popen(
            ["ollama", "serve"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Start a thread to read and display Ollama logs
        def read_logs():
            for line in process.stdout:
                logger.info(f"[Ollama] {line.strip()}")
        
        log_thread = threading.Thread(target=read_logs, daemon=True)
        log_thread.start()
        
        # Wait for Ollama to start
        for attempt in range(1, 11):
            logger.info(f"Waiting for Ollama to start (attempt {attempt}/10)...")
            if check_ollama_running():
                logger.info("Ollama server is running")
                return process
            time.sleep(1)
        
        logger.warning("Ollama did not start within expected time. Continuing anyway...")
        return process
    except FileNotFoundError:
        logger.error("Ollama executable not found. Please install Ollama first.")
        sys.exit(1)

def get_local_ip():
    """Get the local IP address for network access"""
    try:
        # Create a socket that connects to an external server to determine local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # We don't actually need to send data
        s.connect(('8.8.8.8', 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception as e:
        logger.warning(f"Could not determine local IP: {e}")
        return "0.0.0.0"

def prepare_certificates(cert_dir):
    """Prepare SSL certificates"""
    cert_path = Path(cert_dir) / "cert.pem"
    key_path = Path(cert_dir) / "key.pem"
    config_path = Path(cert_dir) / "openssl.cnf"
    
    # Create certificate directory if it doesn't exist
    os.makedirs(cert_dir, exist_ok=True)
    
    # Check if we need to generate certificates
    if not cert_path.exists() or not key_path.exists():
        logger.info("Generating SSL certificates...")
        
        # Create a minimal OpenSSL config with SAN entries
        local_ip = get_local_ip()
        with open(config_path, 'w') as f:
            f.write(f"""
[req]
distinguished_name = req_distinguished_name
x509_extensions = v3_req
prompt = no

[req_distinguished_name]
CN = localhost

[v3_req]
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
IP.1 = 127.0.0.1
IP.2 = {local_ip}
            """)
        
        cmd = [
            "openssl", "req", "-x509", 
            "-newkey", "rsa:4096", 
            "-sha256", 
            "-days", "365", 
            "-nodes", 
            "-keyout", str(key_path), 
            "-out", str(cert_path),
            "-config", str(config_path)
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Certificates generated at {cert_dir}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to generate certificates: {e.stderr.decode()}")
            sys.exit(1)
    else:
        logger.info(f"Using existing certificates from {cert_dir}")
        
    return cert_path, key_path

def run_server(port, cert_dir, auto_start, dev_mode):
    """Start the proxy server"""
    # Prepare certificates
    cert_path, key_path = prepare_certificates(cert_dir)
    
    # Start Ollama if not running and auto_start is enabled
    if auto_start and not check_ollama_running():
        start_ollama_server()
    elif not check_ollama_running():
        logger.warning("Ollama is not running. Proxy may not work until Ollama server is available.")
    else:
        logger.info("Ollama is already running")
    
    # Create server
    class CustomThreadingTCPServer(socketserver.ThreadingTCPServer):
        def __init__(self, *args, **kwargs):
            self.dev_mode = dev_mode
            super().__init__(*args, **kwargs)
    
    httpd = CustomThreadingTCPServer(("", port), OllamaProxy)
    
    # Configure SSL
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(certfile=cert_path, keyfile=key_path)
    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
    
    # Setup simplified shutdown
    def signal_handler(sig, frame):
        logger.info("Shutting down...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Get local IP for network access
    local_ip = get_local_ip()
    
    # Display server information in Vite-like format
    print("\n\033[1m OLLAMA-PROXY \033[0m ready")
    print(f"  ➜  \033[36mLocal:   \033[0mhttps://localhost:{port}/")
    print(f"  ➜  \033[36mNetwork: \033[0mhttps://{local_ip}:{port}/")
    print("\n  Use the Network URL when accessing from another machine\n")
    
    # Start server
    httpd.serve_forever()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Ollama HTTPS Proxy Server")
    parser.add_argument("--port", type=int, default=3838, help="Port to run the proxy server on")
    parser.add_argument("--cert-dir", default="./certs", help="Directory to store certificates")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--dev", action="store_true", help="Development mode (allows all origins)")
    parser.add_argument("--no-start", action="store_true", help="Don't automatically start Ollama if not running")
    
    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    run_server(args.port, args.cert_dir, not args.no_start, args.dev)

if __name__ == "__main__":
    main()
