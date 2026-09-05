from pathlib import Path
from PIL import Image
import xml.etree.ElementTree as ET

BASE = Path("datasets/XF53")
IMAGE_DIR = BASE / "images"
XML_DIR = BASE / "xml"
SPLIT_DIR = BASE / "splits"

NS = {
    "p": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
}

SETS = {
    "train": "training_names.lst",
    "val": "validation_names.lst",
}


def extract_split(split_name, split_file_name):
    split_file = SPLIT_DIR / split_file_name
    out_dir = BASE / split_name
    out_dir.mkdir(exist_ok=True)

    names = [
        x.strip()
        for x in split_file.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]

    count = 0

    for name in names:
        page_name, line_id = name.rsplit("_", 1)

        image_path = IMAGE_DIR / f"{page_name}.png"
        xml_path = XML_DIR / f"{page_name}.xml"

        root = ET.parse(xml_path).getroot()

        line = None
        for candidate in root.findall(".//p:TextLine", NS):
            if candidate.attrib["id"] == line_id:
                line = candidate
                break

        if line is None:
            raise RuntimeError(f"Could not find {line_id} in {xml_path}")

        coords = line.find("p:Coords", NS).attrib["points"]

        points = [
            tuple(map(int, point.split(",")))
            for point in coords.split()
        ]

        xs = [x for x, y in points]
        ys = [y for x, y in points]

        image = Image.open(image_path)

        left = min(xs)
        right = max(xs)
        top = max(0, min(ys) - 2)
        bottom = min(image.height, max(ys) + 2)

        crop = image.crop((left, top, right, bottom))

        text = line.find("p:TextEquiv/p:Unicode", NS).text or ""

        crop.save(out_dir / f"{name}.png")
        (out_dir / f"{name}.gt.txt").write_text(
            text + "\n",
            encoding="utf-8",
        )

        count += 1

    print(f"{split_name}: extracted {count} lines")
    print(f"Output directory: {out_dir}")


def main():
    for split_name, split_file_name in SETS.items():
        extract_split(split_name, split_file_name)


if __name__ == "__main__":
    main()