"""End-to-end API test — uploads test_image.jpg and prints analysis results."""
import json
import sys
from pathlib import Path

import requests

API_URL = "http://localhost:8001"
TEST_IMAGE = Path(__file__).parent / "test_image.jpg"


def main():
    if not TEST_IMAGE.exists():
        print(f"ERROR: Test image not found: {TEST_IMAGE}")
        sys.exit(1)

    print("1. Health check...")
    health = requests.get(f"{API_URL}/health", timeout=10)
    health.raise_for_status()
    print(f"   {health.json()}")

    print("\n2. Analyzing image via POST /analyze...")
    with open(TEST_IMAGE, "rb") as f:
        response = requests.post(
            f"{API_URL}/analyze",
            files={"file": ("test_image.jpg", f, "image/jpeg")},
            timeout=600,
        )

    if response.status_code != 200:
        print(f"ERROR: API error ({response.status_code}): {response.text}")
        sys.exit(1)

    result = response.json()
    print("\nAnalysis complete.\n")
    print("=" * 60)
    print("  MEDIA PROMPTER — ANALYSIS RESULTS")
    print("=" * 60)

    caption = result.get("caption", {}).get("text", "N/A")
    print(f'\nPrompt:\n   "{caption}"')

    print("\nYOLOv8 Detections:")
    dets = result.get("detections", [])
    for d in dets:
        print(
            f"   - {d.get('label', '').capitalize()}: "
            f"{d.get('confidence')}% ({d.get('position')})"
        )
    if not dets:
        print("   No objects detected above threshold.")

    print("\nCLIP Semantic Analysis:")
    print(f"   Mood: {result.get('mood')}")
    print(f"   Scene Type: {result.get('scene_type')}")
    print("   Top Tags:")
    for tag in result.get("semantic_tags", [])[:3]:
        print(f"    - {tag['label']} ({tag['confidence']}%)")

    print("\nEfficientNet Scene Classification:")
    for s in result.get("scene_classifications", [])[:3]:
        print(f"   - {s['label']} ({s['confidence']}%)")

    print("\n" + "=" * 60)

    labels = {d["label"].lower() for d in dets}
    if "dog" not in labels:
        print("\nWarning: expected 'dog' in detections for test_image.jpg")
        sys.exit(1)

    print("\nDog detected — output looks correct.")
    return result


if __name__ == "__main__":
    main()
