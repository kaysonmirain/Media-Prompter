"""Generate a minimal PNG app icon for macOS."""
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "app-icon.png"
OUT.parent.mkdir(exist_ok=True)

size = 1024
img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

radius = int(size * 0.22)
margin = int(size * 0.08)
draw.rounded_rectangle(
    (margin, margin, size - margin, size - margin),
    radius=radius,
    fill=(16, 16, 24, 255),
)

line_color = (167, 139, 250, 255)
line_w = int(size * 0.055)
x1 = int(size * 0.28)
x2 = int(size * 0.72)
x3 = int(size * 0.53)
ys = [int(size * 0.34), int(size * 0.50), int(size * 0.66)]

for y in ys[:2]:
    draw.rounded_rectangle(
        (x1, y - line_w // 2, x2, y + line_w // 2),
        radius=line_w // 2,
        fill=line_color,
    )
draw.rounded_rectangle(
    (x1, ys[2] - line_w // 2, x3, ys[2] + line_w // 2),
    radius=line_w // 2,
    fill=line_color,
)

img.save(OUT)
print(f"Saved {OUT}")
