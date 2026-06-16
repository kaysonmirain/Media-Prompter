import urllib.request
from PIL import Image
import json
import time

print("1. Downloading test image of a dog...")
url = "https://images.unsplash.com/photo-1517849845537-4d257902454a?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80"
path = "test_dog.jpg"
urllib.request.urlretrieve(url, path)

print("2. Initializing Media Prompter (loading models — this takes ~30s)...")
start = time.time()
import sys
sys.path.append('backend')
from backend.analyzer import VisionAnalyzer
analyzer = VisionAnalyzer()
print(f"   Models loaded in {time.time() - start:.1f}s.")

print("3. Analyzing Image...")
img = Image.open(path)
result = analyzer.analyze_image(img)

print("\n" + "="*60)
print("  MEDIA PROMPTER — ANALYSIS RESULTS")
print("="*60)

caption = result.get("caption", {}).get("text", "N/A")
print(f"\nBLIP Caption:\n   \"{caption}\"")

print("\nYOLOv8 Detections:")
dets = result.get("detections", [])
for d in dets:
    print(f"   - {d.get('label', '').capitalize()}: {d.get('confidence')}% ({d.get('position')})")

print("\nCLIP Semantic Analysis:")
print(f"   Mood: {result.get('mood')}")
print(f"   Scene Type: {result.get('scene_type')}")
print("   Top Tags:")
for tag in result.get("semantic_tags", [])[:3]:
    print(f"    - {tag['label']} ({tag['confidence']}%)")

print("\nEfficientNet Scene Classification:")
for s in result.get("scene_classifications", [])[:3]:
    print(f"   - {s['label']} ({s['confidence']}%)")

print("\n" + "="*60)
