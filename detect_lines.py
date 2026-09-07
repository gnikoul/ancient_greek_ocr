from pathlib import Path
import json
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw
from kraken import blla


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PAGE_NAME = "15_01_0053_0015_f_7v_res.png"
# PAGE_NAME = "15_01_0053_0006_f_3r_res.png"

IMAGE_PATH = Path("datasets/XF53/images") / PAGE_NAME
XML_PATH = Path("datasets/XF53/xml") / PAGE_NAME.replace(".png", ".xml")

OUTPUT_IMAGE = Path("detected_boxes.png")
OUTPUT_JSON = Path("detected_boxes.json")


# ---------------------------------------------------------------------
# PAGE XML
# ---------------------------------------------------------------------

PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"


def load_gt_boxes(xml_path):
    root = ET.parse(xml_path).getroot()

    boxes = []

    for text_line in root.iter(f"{{{PAGE_NS}}}TextLine"):
        coords = text_line.find(f"{{{PAGE_NS}}}Coords")

        if coords is None:
            continue

        points = coords.attrib["points"].split()

        xy = []
        for point in points:
            x, y = point.split(",")
            xy.append((int(x), int(y)))

        x_min = min(x for x, y in xy)
        y_min = min(y for x, y in xy)
        x_max = max(x for x, y in xy)
        y_max = max(y for x, y in xy)

        boxes.append({
            "id": text_line.attrib.get("id"),
            "bbox": [x_min, y_min, x_max, y_max],
        })

    return boxes


# ---------------------------------------------------------------------
# Kraken boundary -> bounding box
# ---------------------------------------------------------------------

def boundary_to_bbox(boundary):
    """
    Convert Kraken's line boundary polygon/point list into:

        [x_min, y_min, x_max, y_max]
    """

    xs = [point[0] for point in boundary]
    ys = [point[1] for point in boundary]

    return [
        min(xs),
        min(ys),
        max(xs),
        max(ys),
    ]


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    image = Image.open(IMAGE_PATH).convert("RGB")

    print(f"Image: {IMAGE_PATH}")
    print(f"Size:  {image.size}")
    print()

    # -------------------------------------------------------------
    # Kraken line detection
    # -------------------------------------------------------------

    segmentation = blla.segment(
        image,
        text_direction="horizontal-lr",
        device="cuda",
        autocast=True,
    )
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)

    for i, line in enumerate(segmentation.lines, start=1):
        draw.line(
            line.baseline,
            fill="red",
            width=4,
        )

        draw.text(
            line.baseline[0],
            str(i),
            fill="red",
        )

    image.save("baselines.png")
    predicted_boxes = []

    for line in segmentation.lines:
        bbox = boundary_to_bbox(line.boundary)

        x_min, y_min, x_max, y_max = bbox

        width = x_max - x_min
        height = y_max - y_min

        predicted_boxes.append({
            "bbox": bbox,
            "width": width,
            "height": height,
            "baseline": line.baseline,
        })

    # -------------------------------------------------------------
    # Dynamic filtering
    #
    # Remove tiny detections and detections outside the dominant
    # manuscript text column.
    # -------------------------------------------------------------

    # Remove very small detections.
    filtered = [
        line
        for line in predicted_boxes
        if line["width"] > 0.5 * sorted(
            x["width"] for x in predicted_boxes
        )[len(predicted_boxes) // 2]
    ]

    if not filtered:
        raise RuntimeError("No lines remain after filtering.")

    # Dominant horizontal text region.
    #
    # Instead of hardcoding x coordinates, estimate the dominant
    # manuscript column from the detected lines themselves.
    median_x_min = sorted(
        line["bbox"][0] for line in filtered
    )[len(filtered) // 2]

    median_x_max = sorted(
        line["bbox"][2] for line in filtered
    )[len(filtered) // 2]

    column_width = median_x_max - median_x_min

    filtered = [
        line
        for line in filtered
        if (
            line["bbox"][2] - line["bbox"][0]
            >= 0.5 * column_width
        )
    ]

    # Sort top -> bottom.
    filtered.sort(key=lambda line: line["bbox"][1])

    predicted_boxes = filtered

    print(f"Detected lines after filtering: {len(predicted_boxes)}")
    print()

    # -------------------------------------------------------------
    # Ground truth
    # -------------------------------------------------------------

    gt_boxes = load_gt_boxes(XML_PATH)

    print(f"Ground-truth lines: {len(gt_boxes)}")
    print()

    # -------------------------------------------------------------
    # Padding
    #
    # Horizontal padding only for now.
    # 2 * detected line height on each side.
    # -------------------------------------------------------------

    for line in predicted_boxes:

        x_min, y_min, x_max, y_max = line["bbox"]

        height = y_max - y_min

        padding = 2 * height

        padded_bbox = [
            max(0, x_min - padding),
            y_min,
            min(image.width, x_max + padding),
            y_max,
        ]

        line["padded_bbox"] = padded_bbox

    # -------------------------------------------------------------
    # Print comparison
    # -------------------------------------------------------------

    print(
        "Line  "
        "GT bbox                         "
        "Pred bbox                       "
        "Padded bbox"
    )

    print("-" * 110)

    for i, (pred, gt) in enumerate(
        zip(predicted_boxes, gt_boxes),
        start=1,
    ):

        print(
            f"{i:4d}  "
            f"{str(gt['bbox']):30s} "
            f"{str(pred['bbox']):30s} "
            f"{str(pred['padded_bbox'])}"
        )

    # -------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------

    draw = ImageDraw.Draw(image)

    # GT = blue
    for gt in gt_boxes:
        draw.rectangle(
            gt["bbox"],
            outline="blue",
            width=3,
        )

    # Predicted = red
    for i, pred in enumerate(predicted_boxes, start=1):

        x_min, y_min, x_max, y_max = pred["bbox"]

        draw.rectangle(
            [x_min, y_min, x_max, y_max],
            outline="red",
            width=3,
        )

        draw.text(
            (x_min, y_min),
            str(i),
            fill="red",
        )

        # Padded = green
        draw.rectangle(
            pred["padded_bbox"],
            outline="green",
            width=2,
        )

    image.save(OUTPUT_IMAGE)

    # -------------------------------------------------------------
    # JSON
    # -------------------------------------------------------------

    output = {
        "image": str(IMAGE_PATH),
        "predicted_lines": predicted_boxes,
        "ground_truth_lines": gt_boxes,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print()
    print(f"Saved visualization: {OUTPUT_IMAGE}")
    print(f"Saved boxes:         {OUTPUT_JSON}")


if __name__ == "__main__":
    main()