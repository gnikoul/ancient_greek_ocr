from pathlib import Path
import shutil
import os
import statistics
import xml.etree.ElementTree as ET


PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"

ET.register_namespace("", PAGE_NS)


DATASETS = {
    "eparchos": {
        "xml_dir": Path("datasets/eparchos/xml"),
        "image_dir": Path("datasets/eparchos/image"),
        "output_dir": Path("datasets/segmentation/eparchos"),
    },
    "xf53": {
        "xml_dir": Path("datasets/XF53/xml"),
        "image_dir": Path("datasets/XF53/images"),
        "output_dir": Path("datasets/segmentation/xf53"),
    },
    "xf114": {
        "xml_dir": Path("datasets/XF114/xml"),
        "image_dir": Path("datasets/XF114/images"),
        "output_dir": Path("datasets/segmentation/xf114"),
    },
    "xf79": {
        "xml_dir": Path("datasets/XF79/xml"),
        "image_dir": Path("datasets/XF79/images"),
        "output_dir": Path("datasets/segmentation/xf79"),
    },
}


def add_baselines(xml_path, output_path, image_dir):

    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    page = root.find(f"{{{PAGE_NS}}}Page")

    if page is None:
        raise RuntimeError(f"No Page element found in {xml_path}")

    image_filename = page.attrib["imageFilename"]
    image_path = image_dir / image_filename

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found for {xml_path}: {image_path}"
        )

    page.set(
        "imageFilename",
        os.path.relpath(
            image_path,
            output_path.parent
        )
    )

    text_lines = list(
        root.iter(f"{{{PAGE_NS}}}TextLine")
    )

    added = 0
    skipped = 0

    for line in text_lines:

        # ---------------------------------------------------------
        # Get line coordinates
        # ---------------------------------------------------------

        coords = line.find(
            f"{{{PAGE_NS}}}Coords"
        )

        if coords is None:
            skipped += 1
            continue

        line_points = [
            tuple(map(int, point.split(",")))
            for point in coords.attrib["points"].split()
        ]

        x_min = min(x for x, y in line_points)
        x_max = max(x for x, y in line_points)

        # ---------------------------------------------------------
        # Get word bottoms
        # ---------------------------------------------------------

        word_bottoms = []

        for word in line.findall(
            f"{{{PAGE_NS}}}Word"
        ):

            word_coords = word.find(
                f"{{{PAGE_NS}}}Coords"
            )

            if word_coords is None:
                continue

            word_points = [
                tuple(map(int, point.split(",")))
                for point in word_coords.attrib["points"].split()
            ]

            word_y_max = max(
                y for x, y in word_points
            )

            word_bottoms.append(word_y_max)

        # ---------------------------------------------------------
        # We need at least one word to determine the baseline.
        # ---------------------------------------------------------

        if not word_bottoms:
            skipped += 1
            continue

        baseline_y = int(
            statistics.median(word_bottoms)
        )

        # ---------------------------------------------------------
        # Remove an existing baseline if one happens to exist.
        # ---------------------------------------------------------

        old_baseline = line.find(
            f"{{{PAGE_NS}}}Baseline"
        )

        if old_baseline is not None:
            line.remove(old_baseline)

        # ---------------------------------------------------------
        # Create baseline.
        # ---------------------------------------------------------

        baseline = ET.Element(
            f"{{{PAGE_NS}}}Baseline"
        )

        baseline.set(
            "points",
            f"{x_min},{baseline_y} "
            f"{x_max},{baseline_y}"
        )

        # Put Baseline immediately after Coords.
        coords_index = list(line).index(coords)

        line.insert(
            coords_index + 1,
            baseline
        )

        added += 1

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    tree.write(
        output_path,
        encoding="utf-8",
        xml_declaration=True
    )

    return len(text_lines), added, skipped


def main():

    total_lines = 0
    total_added = 0
    total_skipped = 0

    print("=" * 70)
    print("Creating segmentation training PAGE XML")
    print("=" * 70)

    for name, config in DATASETS.items():

        xml_dir = config["xml_dir"]
        output_dir = config["output_dir"]

        xml_files = sorted(
            xml_dir.glob("*.xml")
        )

        print()
        print(name.upper())
        print("-" * 70)
        print(f"Input XML files:  {len(xml_files)}")
        print(f"Output directory: {output_dir}")

        dataset_lines = 0
        dataset_added = 0
        dataset_skipped = 0

        for xml_path in xml_files:

            output_path = (
                output_dir / xml_path.name
            )

            lines, added, skipped = add_baselines(
                xml_path,
                output_path,
                config["image_dir"]
            )

            dataset_lines += lines
            dataset_added += added
            dataset_skipped += skipped

        print(f"Lines:            {dataset_lines}")
        print(f"Baselines added:  {dataset_added}")
        print(f"Lines skipped:    {dataset_skipped}")

        total_lines += dataset_lines
        total_added += dataset_added
        total_skipped += dataset_skipped

    print()
    print("=" * 70)
    print("TOTAL")
    print("=" * 70)
    print(f"Lines:            {total_lines}")
    print(f"Baselines added:  {total_added}")
    print(f"Lines skipped:    {total_skipped}")


if __name__ == "__main__":
    main()