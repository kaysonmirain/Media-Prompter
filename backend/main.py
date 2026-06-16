"""
Media Prompter FastAPI Backend
Serves the analysis API with WebSocket progress streaming.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import torch  # MUST BE FIRST on macOS to avoid MPS/SSL deadlocks
import asyncio
import json
import os
import numpy as np
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

# ── Initialize App ────────────────────────────────────────────────────────────
app = FastAPI(
    title="Media Prompter API",
    description="Descriptions and deep analysis for images and videos",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_IMAGE_TYPES = {
    "image/jpeg", "image/png", "image/gif", "image/bmp",
    "image/webp", "image/tiff",
}
ALLOWED_VIDEO_TYPES = {
    "video/mp4", "video/mpeg", "video/quicktime", "video/x-msvideo",
    "video/x-matroska", "video/webm", "video/x-flv",
}

# ── Load models synchronously on main thread at startup ──────────────────────
# This completely prevents any macOS PyTorch MPS threading deadlocks.
import sys
sys.path.insert(0, str(Path(__file__).parent))
print("Initializing Media Prompter models... (takes ~15 seconds)")
try:
    from analyzer import VisionAnalyzer
    from video_processor import VideoProcessor
    _analyzer = VisionAnalyzer()
    _video_processor = VideoProcessor(_analyzer, max_frames=16, fps_target=0.5)
    print("Models initialized successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    _analyzer = None
    _video_processor = None

def get_analyzer():
    if _analyzer is None:
        raise HTTPException(status_code=500, detail="Models failed to load at startup.")
    return _analyzer

def get_video_processor():
    if _video_processor is None:
        raise HTTPException(status_code=500, detail="Models failed to load at startup.")
    return _video_processor


def _json_safe(obj):
    """Convert numpy/torch scalars to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return _json_safe(obj.detach().cpu().tolist())
    return obj


def detect_media_type(filename: str, content_type: str) -> str:
    """Detect if file is image or video based on extension and MIME type."""
    ext = Path(filename).suffix.lower()
    image_exts = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"}
    video_exts = {".mp4", ".mpeg", ".mov", ".avi", ".mkv", ".webm", ".flv", ".m4v"}

    if ext in image_exts or content_type in ALLOWED_IMAGE_TYPES:
        return "image"
    elif ext in video_exts or content_type in ALLOWED_VIDEO_TYPES:
        return "video"
    else:
        return "unknown"


@app.post("/analyze")
async def analyze_media(file: UploadFile = File(...)):
    """
    Upload an image or video file for analysis.
    Processes synchronously on the main thread to avoid macOS PyTorch deadlocks.
    """
    # Save uploaded file
    suffix = Path(file.filename or "upload").suffix or ".jpg"
    task_id = str(uuid.uuid4())
    upload_path = UPLOAD_DIR / f"{task_id}{suffix}"

    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        from PIL import Image as PILImage
        media_type = "image" if suffix.lower() in [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".gif"] else "video"

        if media_type == "image":
            analyzer = get_analyzer()
            img = PILImage.open(str(upload_path)).convert("RGB")
            # We don't need progress callbacks since we are doing a simple blocking request
            result = analyzer.analyze_image(img, None)
            result["type"] = "image"
            result["filename"] = file.filename

        else:  # video
            vp = get_video_processor()
            result = vp.process_video(str(upload_path), None)
            result["filename"] = file.filename

        return JSONResponse(_json_safe(result))

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(upload_path)
        except Exception:
            pass


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    import torch
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    return {
        "status": "healthy",
        "models_loaded": _analyzer is not None,
        "device": device,
        "pytorch_version": torch.__version__
    }


# Removed preload endpoints to prevent macOS PyTorch ThreadPool deadlocks


# Serve frontend
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    print("=" * 60)
    print("  Media Prompter — Deep Vision Analysis System")
    print("  Media Analysis")
    print("=" * 60)
    port = int(os.environ.get("PORT", "8001"))
    print(f"  Server: http://localhost:{port}")
    print(f"  API Docs: http://localhost:{port}/docs")
    print("=" * 60)

    port = int(os.environ.get("PORT", "8001"))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
