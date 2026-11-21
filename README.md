# Media Prompter

Upload an image or video and get a descriptive prompt back — captions, detected objects, scene type, mood, and semantic tags.

## Setup

Python 3.9+. From the project folder:

```bash
chmod +x start.sh
./start.sh
```

Starts at http://localhost:8001. The first run installs dependencies into `.venv` and downloads several model weights. Wait for the terminal to report that the models loaded. The `/health` endpoint returns HTTP 503 if initialization fails, and the terminal shows the startup error.

With the server running, use a JPEG of a dog to check the upload and object detection.

```bash
.venv/bin/python test_api.py /path/to/dog.jpg
```

## How it's built

The backend is FastAPI. On startup, `backend/main.py` loads YOLOv8x for object detection, BLIP for captions, CLIP ViT-L/14 for tags and mood, and EfficientNet-B7 for scene labels.

When you upload a file, the API returns a prompt built from caption text, detections, scene scores, and CLIP outputs.

Videos go through `video_processor.py`, which samples up to 16 frames and runs the same pipeline on each one, then rolls it into a timeline summary.

The frontend is HTML, CSS, and JavaScript in `frontend/`, with drag-and-drop upload and tabs for detections, scene breakdown, mood, tags, and raw JSON.

`start.sh` sets up the venv, installs from `requirements.txt`, and launches uvicorn. The server also serves the static frontend files.
