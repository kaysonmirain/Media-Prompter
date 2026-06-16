# Media Prompter

Drop in an image or video and get a prompt back — captions, what objects are in the frame, scene type, mood, and a handful of semantic tags. Handy when you need a solid text description of visual media without writing it yourself.

**Kayson Mirain**

## Setup

You need Python 3.9 or newer. From the project folder:

```bash
chmod +x start.sh
./start.sh
```

That creates a venv, installs dependencies, and starts the server at http://localhost:6666. The browser opens on its own. First launch takes a bit while models download.

## Models

- **BLIP** — captions and the main prompt text
- **YOLOv8x** — object detection
- **CLIP ViT-L/14** — tags, mood, scene semantics
- **EfficientNet-B7** — scene classification

Videos are sampled frame-by-frame (up to 16 frames) and summarized into a timeline.
