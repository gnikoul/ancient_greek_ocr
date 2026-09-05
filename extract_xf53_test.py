from pathlib import Path
from PIL import Image
import xml.etree.ElementTree as ET


BASE = Path("datasets/XF53")

IMAGE_DIR = BASE / "images"
XML_DIR = BASE / "xml"
SPLIT_FILE = BASE / "splits/testing_names.lst"
OUT_DIR = BASE / "test"

NS = {
    "p": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
}


def main():
    OUT_DIR.mkdir(exist_ok=True)

    names = [
        x.strip()
        for x in SPLIT_FILE.read_text(encoding="utf-8").splitlines()
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
            raise RuntimeError(
                f"Could not find {line_id} in {xml_path}"
            )

        coords = line.find("p:Coords", NS).attrib["points"]

        points = [
            tuple(map(int, point.split(",")))
            for point in coords.split()
        ]

        xs = [x for x, y in points]
        ys = [y for x, y in points]

        image = Image.open(image_path)

        # Add a small vertical margin to avoid clipping characters.
        left = min(xs)
        right = max(xs)
        top = max(0, min(ys) - 2)
        bottom = min(image.height, max(ys) + 2)

        crop = image.crop((left, top, right, bottom))

        text = line.find("p:TextEquiv/p:Unicode", NS).text or ""

        output_stem = name

        crop.save(OUT_DIR / f"{output_stem}.png")

        (OUT_DIR / f"{output_stem}.gt.txt").write_text(
            text + "\n",
            encoding="utf-8"
        )

        count += 1

    print(f"Extracted {count} test lines")
    print(f"Output directory: {OUT_DIR}")


if __name__ == "__main__":
    main()