from pathlib import Path
import json
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw
from kraken import blla
from kraken.models import load_models


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PAGE_NAME = "15_01_0053_0015_f_7v_res.png"
# PAGE_NAME = "15_01_0053_0006_f_3r_res.png"

IMAGE_PATH = Path("datasets/XF53/images") / PAGE_NAME
XML_PATH = Path("datasets/XF53/xml") / PAGE_NAME.replace(".png", ".xml")

BASE_MODEL = Path(
    "/home/gnikoul/miniconda3/envs/grc/lib/python3.11/"
    "site-packages/kraken/blla.mlmodel"
)

FINETUNED_MODEL = Path(
    "models/segmentation_finetuned/best_0.5324.safetensors"
)

OUTPUT_IMAGE = Path("segmentation_comparison.png")
OUTPUT_BEFORE_JSON = Path("segmentation_before.json")
OUTPUT_AFTER_JSON = Path("segmentation_after.json")


PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"


# ---------------------------------------------------------------------
# PAGE XML
# ---------------------------------------------------------------------

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

    xs = [point[0] for point in boundary]
    ys = [point[1] for point in boundary]

    return [
        min(xs),
        min(ys),
        max(xs),
        max(ys),
    ]


# ---------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------

def detect_lines(image, model_path):

    print(f"Loading model: {model_path}")


    segmentation_model = load_models(str(model_path))[0]
    
    if "hyper_params" not in segmentation_model.user_metadata:
        base_model = load_models(str(BASE_MODEL))[0]
        segmentation_model.user_metadata["hyper_params"] = (
            base_model.user_metadata["hyper_params"]
        )


    segmentation = blla.segment(
        image,
        text_direction="horizontal-lr",
        device="cuda",
        model=segmentation_model,
        autocast=True,
    )

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
    # -------------------------------------------------------------

    median_width = sorted(
        x["width"] for x in predicted_boxes
    )[len(predicted_boxes) // 2]

    filtered = [
        line
        for line in predicted_boxes
        if line["width"] > 0.5 * median_width
    ]

    if not filtered:
        raise RuntimeError("No lines remain after filtering.")

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

    filtered.sort(key=lambda line: line["bbox"][1])

    return filtered


# ---------------------------------------------------------------------
# Draw result
# ---------------------------------------------------------------------

def draw_result(image, predicted_boxes, gt_boxes, title):

    image = image.copy()

    draw = ImageDraw.Draw(image)

    # GT = blue
    for gt in gt_boxes:

        draw.rectangle(
            gt["bbox"],
            outline="blue",
            width=3,
        )

    # Prediction = red
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

    # Title
    draw.rectangle(
        [0, 0, image.width, 45],
        fill="white",
    )

    draw.text(
        (10, 10),
        title,
        fill="black",
    )

    return image


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    image = Image.open(IMAGE_PATH).convert("RGB")

    print(f"Image: {IMAGE_PATH}")
    print(f"Size:  {image.size}")
    print()

    # -------------------------------------------------------------
    # Ground truth
    # -------------------------------------------------------------

    gt_boxes = load_gt_boxes(XML_PATH)

    print(f"Ground-truth lines: {len(gt_boxes)}")
    print()

    # -------------------------------------------------------------
    # BEFORE
    # -------------------------------------------------------------

    before_boxes = detect_lines(
        image,
        BASE_MODEL,
    )

    print(
        f"Before: {len(before_boxes)} detected lines"
    )

    # -------------------------------------------------------------
    # AFTER
    # -------------------------------------------------------------

    after_boxes = detect_lines(
        image,
        FINETUNED_MODEL,
    )

    print(
        f"After:  {len(after_boxes)} detected lines"
    )

    # -------------------------------------------------------------
    # Print comparison
    # -------------------------------------------------------------

    print()
    print(
        "Line  "
        "GT bbox                         "
        "Before bbox                    "
        "After bbox"
    )

    print("-" * 110)

    for i, gt in enumerate(gt_boxes):

        before = (
            before_boxes[i]["bbox"]
            if i < len(before_boxes)
            else None
        )

        after = (
            after_boxes[i]["bbox"]
            if i < len(after_boxes)
            else None
        )

        print(
            f"{i + 1:4d}  "
            f"{str(gt['bbox']):30s} "
            f"{str(before):30s} "
            f"{str(after)}"
        )

    # -------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------

    before_image = draw_result(
        image,
        before_boxes,
        gt_boxes,
        "BEFORE - pretrained BLLA",
    )

    after_image = draw_result(
        image,
        after_boxes,
        gt_boxes,
        "AFTER - fine-tuned BLLA",
    )

    comparison = Image.new(
        "RGB",
        (
            before_image.width * 2,
            before_image.height,
        ),
        "white",
    )

    comparison.paste(
        before_image,
        (0, 0),
    )

    comparison.paste(
        after_image,
        (before_image.width, 0),
    )

    comparison.save(OUTPUT_IMAGE)

    # -------------------------------------------------------------
    # JSON
    # -------------------------------------------------------------

    with open(
        OUTPUT_BEFORE_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "image": str(IMAGE_PATH),
                "model": str(BASE_MODEL),
                "predicted_lines": before_boxes,
                "ground_truth_lines": gt_boxes,
            },
            f,
            indent=2,
        )

    with open(
        OUTPUT_AFTER_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "image": str(IMAGE_PATH),
                "model": str(FINETUNED_MODEL),
                "predicted_lines": after_boxes,
                "ground_truth_lines": gt_boxes,
            },
            f,
            indent=2,
        )

    print()
    print(f"Saved comparison: {OUTPUT_IMAGE}")
    print(f"Saved before:     {OUTPUT_BEFORE_JSON}")
    print(f"Saved after:      {OUTPUT_AFTER_JSON}")


if __name__ == "__main__":
    main()