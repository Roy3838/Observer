from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import argparse
import logging
import os
import socket
import subprocess
from pathlib import Path

# Import routers from our modules
from marketplace import marketplace_router
from compute import compute_router
from tools_router import tools_router
from payments import payments_router

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('api-server')

# modelo = randomforest()

# @app.post("/resultados")
# def enviar_resultados(datos):
#     resultados = modelo(datos)
#     return resultados

# Setup FastAPI app
app = FastAPI()

# Create temp images directory
TEMP_IMAGES_DIR = Path("temp_images")
TEMP_IMAGES_DIR.mkdir(exist_ok=True)

# Mount static files for serving images
app.mount("/temp-images", StaticFiles(directory="temp_images"), name="temp-images")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers - without prefixes to maintain original URL structure
app.include_router(marketplace_router)
# Mount compute router last since it has a catch-all route
app.include_router(compute_router)
# Mount twilio router
app.include_router(tools_router)
# Payments router
app.include_router(
    payments_router,
    prefix="/payments",
    tags=["Payments"]
)

# Root path to check if service is running
@app.get("/")
async def root():
    return {"status": "API server is running"}

# Generate SSL certificates
def prepare_certificates(cert_dir):
    """Prepare SSL certificates"""
    cert_path = Path(cert_dir) / "cert.pem"
    key_path = Path(cert_dir) / "key.pem"
    
    # Create certificate directory if it doesn't exist
    os.makedirs(cert_dir, exist_ok=True)
    
    # Check if we need to generate certificates
    if not cert_path.exists() or not key_path.exists():
        logger.info("Generating SSL certificates...")
        cmd = [
            "openssl", "req", "-x509", 
            "-newkey", "rsa:4096", 
            "-sha256", 
            "-days", "365", 
            "-nodes", 
            "-keyout", str(key_path), 
            "-out", str(cert_path),
            "-subj", "/CN=localhost"
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Certificates generated at {cert_dir}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to generate certificates: {e.stderr.decode() if hasattr(e, 'stderr') else str(e)}")
            raise RuntimeError("Failed to generate SSL certificates")
    else:
        logger.info(f"Using existing certificates from {cert_dir}")
        
    return cert_path, key_path

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Observer AI API Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run on")
    parser.add_argument("--cert-dir", default="./certs", help="Certificate directory")
    parser.add_argument("--cert-file", help="Path to certificate file (if not provided, self-signed will be used)")
    parser.add_argument("--key-file", help="Path to key file (if not provided, self-signed will be used)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--proxy-target", default="https://compute.observer-ai.com", help="Target service URL for proxy")
    
    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
        
    # Set target URL for compute proxy
    os.environ["AI_SERVICE_URL"] = args.proxy_target
    
    # Set up SSL
    if args.cert_file and args.key_file:
        # Use provided certificates
        cert_path = args.cert_file
        key_path = args.key_file
        logger.info(f"Using provided certificates: {cert_path}, {key_path}")
    else:
        # Self-sign certificates
        cert_path, key_path = prepare_certificates(args.cert_dir)
    
    # Get local IP for network access
    local_ip = get_local_ip()
    
    # Print server info
    print("\n\033[1m OBSERVER AI API SERVER \033[0m ready")
    print(f"  ➜  \033[36mLocal:   \033[0mhttps://localhost:{args.port}/")
    print(f"  ➜  \033[36mNetwork: \033[0mhttps://{local_ip}:{args.port}/")
    print(f"\n  Marketplace routes: https://localhost:{args.port}/agents")
    print(f"  Compute quota: https://localhost:{args.port}/quota")
    print(f"  Proxy forwarding to: {args.proxy_target}")
    
    # Run with SSL context
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=args.port,
        ssl_certfile=str(cert_path),
        ssl_keyfile=str(key_path)
    )
