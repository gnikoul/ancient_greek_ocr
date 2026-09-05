from pathlib import Path
from lxml import etree
from PIL import Image

DATASET = Path("datasets/eparchos")
IMAGE_DIR = DATASET / "image"
XML_DIR = DATASET / "xml"
OUTPUT_DIR = DATASET / "samples"

OUTPUT_DIR.mkdir(exist_ok=True)

xml_files = sorted(XML_DIR.glob("*.xml"))

samples = 0

for xml_file in xml_files:
    tree = etree.parse(xml_file)
    lines = tree.xpath('//*[local-name()="TextLine"]')

    image_file = IMAGE_DIR / (xml_file.stem + ".png")

    if not image_file.exists():
        continue

    image = Image.open(image_file)

    for line in lines:
        text = line.xpath(
            './/*[local-name()="TextEquiv"]'
            '/*[local-name()="Unicode"]/text()'
        )

        points = line.xpath('.//*[local-name()="Coords"]')

        if not text or not points:
            continue

        text = text[0]

        coords = points[0].get("points", "")
        points = [
            tuple(map(int, p.split(",")))
            for p in coords.split()
        ]

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        left = max(0, min(xs) - 10)
        top = max(0, min(ys) - 10)
        right = min(image.width, max(xs) + 10)
        bottom = min(image.height, max(ys) + 10)

        crop = image.crop((left, top, right, bottom))

        output = OUTPUT_DIR / f"{samples + 1:02d}.png"
        crop.save(output)

        print(f"{samples + 1:02d}: {text}")
        print(f"    {output}")

        samples += 1

        if samples == 10:
            break

    if samples == 10:
        break