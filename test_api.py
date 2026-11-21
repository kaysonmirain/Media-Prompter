"""End-to-end API test using a JPEG of a dog."""
import argparse
import os
import sys
from pathlib import Path

import requests

TEST_IMAGE = Path(__file__).parent / "test_image.jpg"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, nargs="?", default=TEST_IMAGE)
    parser.add_argument("--url", default=f"http://localhost:{os.environ.get('PORT', '8001')}")
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"Provide the path to a JPEG of a dog. File not found: {args.image}")
    api_url = args.url.rstrip("/")

    print("1. Health check...")
    health = requests.get(f"{api_url}/health", timeout=10)
    health.raise_for_status()
    print(f"   {health.json()}")

    print("\n2. Analyzing image via POST /analyze...")
    with args.image.open("rb") as f:
        response = requests.post(
            f"{api_url}/analyze",
            files={"file": (args.image.name, f, "image/jpeg")},
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
        print(f"\nExpected a dog detection for {args.image.name}.")
        sys.exit(1)

    print("\nDog detected — output looks correct.")
    return result


if __name__ == "__main__":
    main()
