from pathlib import Path
from PIL import Image
from collections import Counter

DATASET = Path("datasets/eparchos/training")
IMAGE_DIR = DATASET / "images"
LABELS_FILE = DATASET / "labels.txt"

labels = []
widths = []
heights = []

with open(LABELS_FILE, encoding="utf-8") as f:
    for line in f:
        filename, text = line.rstrip("\n").split("\t", 1)

        labels.append(text)

        image = Image.open(IMAGE_DIR / filename)
        width, height = image.size

        widths.append(width)
        heights.append(height)


print(f"Samples: {len(labels)}")
print()

print(f"Average line width:  {sum(widths) / len(widths):.1f} px")
print(f"Average line height: {sum(heights) / len(heights):.1f} px")
print(f"Minimum width:       {min(widths)} px")
print(f"Maximum width:       {max(widths)} px")
print(f"Minimum height:      {min(heights)} px")
print(f"Maximum height:      {max(heights)} px")

print()

# Character statistics
characters = Counter("".join(labels))

print(f"Unique characters: {len(characters)}")
print()
print("Characters:")
print("".join(sorted(characters)))

print()
print("Most common characters:")

for char, count in characters.most_common(30):
    print(repr(char), count)