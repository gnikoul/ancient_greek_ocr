from pathlib import Path
from lxml import etree
from PIL import Image


DATASET = Path("datasets/eparchos")
IMAGE_DIR = DATASET / "image"
XML_DIR = DATASET / "xml"

OUTPUT_DIR = DATASET / "training"
OUTPUT_IMAGES = OUTPUT_DIR / "images"
LABELS_FILE = OUTPUT_DIR / "labels.txt"

OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)


def get_text(line):
    text = line.xpath(
        './*[local-name()="TextEquiv"]'
        '/*[local-name()="Unicode"]/text()'
    )
    return text[0].strip() if text else ""


def get_points(line):
    coords = line.xpath('.//*[local-name()="Coords"]')

    if not coords:
        return []

    points = coords[0].get("points", "")

    return [
        tuple(map(int, point.split(",")))
        for point in points.split()
    ]


samples = 0

with open(LABELS_FILE, "w", encoding="utf-8") as labels:

    for xml_file in sorted(XML_DIR.glob("*.xml")):

        image_file = IMAGE_DIR / f"{xml_file.stem}.png"

        if not image_file.exists():
            print(f"WARNING: image not found: {image_file}")
            continue

        image = Image.open(image_file)

        tree = etree.parse(xml_file)
        lines = tree.xpath('//*[local-name()="TextLine"]')

        for line in lines:

            text = get_text(line)
            points = get_points(line)

            if not text or not points:
                continue

            xs = [p[0] for p in points]
            ys = [p[1] for p in points]

            # Small padding around the annotated line
            left = max(0, min(xs) - 10)
            top = max(0, min(ys) - 10)
            right = min(image.width, max(xs) + 10)
            bottom = min(image.height, max(ys) + 10)

            crop = image.crop((left, top, right, bottom))

            filename = f"{samples:06d}.png"
            crop.save(OUTPUT_IMAGES / filename)

            labels.write(f"{filename}\t{text}\n")

            samples += 1

print()
print(f"Created {samples} samples")
print(f"Images: {OUTPUT_IMAGES}")
print(f"Labels: {LABELS_FILE}")