# Media Prompter

Upload an image or video and get a descriptive prompt back — captions, detected objects, scene type, mood, and semantic tags.

## Setup

Python 3.9+. From the project folder:

```bash
chmod +x start.sh
./start.sh
```

Starts at http://localhost:6666. First run installs deps into `.venv` and pulls model weights.

## How it's built

The backend is FastAPI. On startup, `backend/main.py` loads a `VisionAnalyzer` that runs four models in parallel — YOLOv8x for object detection, BLIP for captions, CLIP ViT-L/14 for tags and mood, and EfficientNet-B7 for scene labels. Each model has its own file under `backend/models/`.

When you upload a file, the API queues analysis and streams progress over WebSocket. The analyzer builds the final prompt from caption text, detections, scene scores, and CLIP outputs.

Videos go through `video_processor.py`, which samples up to 16 frames and runs the same pipeline on each one, then rolls it into a timeline summary.

The frontend is vanilla HTML/CSS/JS in `frontend/` — drag-and-drop upload, live progress, and tabs for detections, scene breakdown, mood, tags, and raw JSON. No React or build step.

`start.sh` sets up the venv, installs from `requirements.txt`, and launches uvicorn. The server also serves the static frontend files.
