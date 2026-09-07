import torch
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from kraken_monkey_patch import apply_kraken_monkey_patch

apply_kraken_monkey_patch()

from kraken import blla
from kraken.models import load_models
from kraken.configs import RecognitionInferenceConfig
from kraken.lib.dataset import ImageInputTransforms


# ============================================================
# CONFIGURATION
# ============================================================

# PAGE_NAME = "15_01_0053_0015_f_7v_res.png"
PAGE_NAME = "15_01_0053_0006_f_3r_res.png"

IMAGE_DIR = Path("datasets/XF53/images")
XML_DIR = Path("datasets/XF53/xml")

# ------------------------------------------------------------
# BLLA segmentation model
# ------------------------------------------------------------

SEGMENTATION_MODEL_PATH = Path(
    "models/segmentation_finetuned/"
    "best_0.5324.safetensors"
)

BASE_BLLA_MODEL_PATH = Path(
    "/home/gnikoul/miniconda3/envs/grc/lib/python3.11/"
    "site-packages/kraken/blla.mlmodel"
)

# ------------------------------------------------------------
# Recognition model
# ------------------------------------------------------------

MODEL_PATH = Path(
    "models/combined_ppocrv6/"
    "best_0.8614.safetensors"
)

OUTPUT_IMAGE = Path("ocr_example.png")


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(text):
    return unicodedata.normalize("NFD", text.strip())


# ============================================================
# LOAD RECOGNITION MODEL
# ============================================================

def load_recognition_model():

    print("=" * 70)
    print("LOADING EPARCHOS+XF53 RECOGNITION MODEL")
    print("=" * 70)

    model = load_models(str(MODEL_PATH))[0]

    print("Codec size:", len(model.codec))

    config = RecognitionInferenceConfig(
        accelerator="cuda",
        devices=1,
        batch_size=1,
        padding=0,
    )

    model.prepare_for_inference(config)

    print(
        "Device:",
        next(model.parameters()).device,
    )

    return model


# ============================================================
# LOAD BLLA MODEL
# ============================================================

def load_segmentation_model():

    print("=" * 70)
    print("LOADING FINE-TUNED BLLA MODEL")
    print("=" * 70)

    model = load_models(
        str(SEGMENTATION_MODEL_PATH)
    )[0]

    print(
        "Segmentation model:",
        SEGMENTATION_MODEL_PATH,
    )

    # --------------------------------------------------------
    # Kraken's blla.segment() expects hyper_params metadata.
    #
    # The segtrain-produced safetensors checkpoint does not
    # contain this field, so copy it from the original BLLA
    # model. This only modifies the in-memory model.
    # --------------------------------------------------------

    if "hyper_params" not in model.user_metadata:

        base_model = load_models(
            str(BASE_BLLA_MODEL_PATH)
        )[0]

        model.user_metadata["hyper_params"] = (
            base_model.user_metadata["hyper_params"]
        )

        print("Copied BLLA hyper_params from base model.")

    return model


# ============================================================
# BLLA BOUNDARY -> BBOX
# ============================================================

def boundary_to_bbox(boundary):

    xs = [point[0] for point in boundary]
    ys = [point[1] for point in boundary]

    return [
        min(xs),
        min(ys),
        max(xs),
        max(ys),
    ]


# ============================================================
# DETECT PAGE LINES
# ============================================================

def detect_lines(page_image, segmentation_model):

    print()
    print("=" * 70)
    print("SEGMENTING PAGE WITH BLLA")
    print("=" * 70)

    segmentation = blla.segment(
        page_image,
        text_direction="horizontal-lr",
        device="cuda",
        model=segmentation_model,
        autocast=True,
    )

    predicted_boxes = []

    for line in segmentation.lines:

        bbox = boundary_to_bbox(
            line.boundary
        )

        x_min, y_min, x_max, y_max = bbox

        width = x_max - x_min
        height = y_max - y_min

        predicted_boxes.append({
            "bbox": bbox,
            "width": width,
            "height": height,
            "baseline": line.baseline,
        })

    print(
        f"Raw BLLA detections: "
        f"{len(predicted_boxes)}"
    )

    # --------------------------------------------------------
    # Dynamic filtering
    #
    # Same filtering used in the segmentation comparison.
    # This removes small detections such as material from
    # neighboring pages.
    # --------------------------------------------------------

    median_width = sorted(
        x["width"]
        for x in predicted_boxes
    )[len(predicted_boxes) // 2]

    filtered = [
        line
        for line in predicted_boxes
        if line["width"] > 0.5 * median_width
    ]

    if not filtered:
        raise RuntimeError(
            "No lines remain after width filtering."
        )

    median_x_min = sorted(
        line["bbox"][0]
        for line in filtered
    )[len(filtered) // 2]

    median_x_max = sorted(
        line["bbox"][2]
        for line in filtered
    )[len(filtered) // 2]

    column_width = (
        median_x_max - median_x_min
    )

    filtered = [
        line
        for line in filtered
        if (
            line["bbox"][2] - line["bbox"][0]
            >= 0.5 * column_width
        )
    ]

    # Top-to-bottom reading order.
    filtered.sort(
        key=lambda line: line["bbox"][1]
    )

    print(
        f"Lines after filtering: "
        f"{len(filtered)}"
    )

    return filtered


# ============================================================
# PREDICT ONE LINE
# ============================================================

def predict_line(model, image):

    batch, channels, height, width = model.input

    transforms = ImageInputTransforms(
        batch,
        height,
        width,
        channels,
        (
            model._inf_config.padding,
            0,
        ),
        False,
        dtype=model._m_dtype,
    )

    with model._fabric.init_tensor():

        ts_im = transforms(image)

    seqs = ts_im.unsqueeze(0)

    seq_lens = torch.tensor(
        [ts_im.shape[2]],
        device=seqs.device,
    )

    preds, olens = model._rec_predict(
        seqs,
        seq_lens,
    )

    prediction = "".join(
        x[0]
        for x in preds[0]
    )

    return normalize(prediction)


# ============================================================
# RECOGNIZE BLLA-DETECTED LINES
# ============================================================

def recognize_page(
    model,
    page_image,
    lines,
):

    predictions = []

    print()
    print("=" * 70)
    print("RECOGNIZING BLLA-DETECTED LINES")
    print("=" * 70)

    print(
        f"Lines detected: {len(lines)}"
    )
    print()

    for i, line in enumerate(
        lines,
        1,
    ):

        left, top, right, bottom = (
            line["bbox"]
        )

        # ----------------------------------------------------
        # Small safety padding.
        #
        # This is deliberately small because the BLLA model
        # should already be responsible for finding the line.
        # ----------------------------------------------------

        padding_x = 2
        padding_y = 2

        crop_left = max(
            0,
            left - padding_x,
        )

        crop_top = max(
            0,
            top - padding_y,
        )

        crop_right = min(
            page_image.width,
            right + padding_x,
        )

        crop_bottom = min(
            page_image.height,
            bottom + padding_y,
        )

        crop = page_image.crop(
            (
                crop_left,
                crop_top,
                crop_right,
                crop_bottom,
            )
        )

        prediction = predict_line(
            model,
            crop,
        )

        predictions.append({
            "bbox": line["bbox"],
            "prediction": prediction,
        })

        print(
            f"{i:3d}/{len(lines):3d} "
            f"{line['bbox']} "
            f"{prediction}"
        )

    return predictions


# ============================================================
# FONT
# ============================================================

def find_font(size):

    possible_fonts = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
    ]

    for font_path in possible_fonts:

        path = Path(font_path)

        if path.exists():

            return ImageFont.truetype(
                str(path),
                size,
            )

    return ImageFont.load_default()


# ============================================================
# CREATE DISPLAY IMAGE
# ============================================================

def create_result_image(
    original_page,
    detected_lines,
    predictions,
    output_path,
):

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    page_width = 1000

    scale = (
        page_width
        / original_page.width
    )

    page_height = int(
        original_page.height
        * scale
    )

    page = original_page.resize(
        (
            page_width,
            page_height,
        )
    )

    text_width = 850

    margin = 40

    title_height = 100

    line_height = 65

    transcription_height = max(
        500,
        title_height
        + len(predictions)
        * line_height
        + margin * 2,
    )

    total_width = (
        page_width
        + text_width
        + margin * 3
    )

    total_height = max(
        page_height
        + margin * 2,
        transcription_height,
    )

    canvas = Image.new(
        "RGB",
        (
            total_width,
            total_height,
        ),
        "white",
    )

    draw = ImageDraw.Draw(canvas)

    # --------------------------------------------------------
    # Fonts
    # --------------------------------------------------------

    title_font = find_font(38)
    text_font = find_font(30)

    # --------------------------------------------------------
    # Page
    # --------------------------------------------------------

    page_x = margin

    page_y = (
        margin
        + title_height
    )

    canvas.paste(
        page,
        (
            page_x,
            page_y,
        ),
    )

    # --------------------------------------------------------
    # Draw BLLA bounding boxes
    # --------------------------------------------------------

    for i, line in enumerate(
        detected_lines,
        1,
    ):

        left, top, right, bottom = (
            line["bbox"]
        )

        # Convert original coordinates
        # to display coordinates.

        left = int(
            left * scale
        )

        top = int(
            top * scale
        )

        right = int(
            right * scale
        )

        bottom = int(
            bottom * scale
        )

        left += page_x
        right += page_x

        top += page_y
        bottom += page_y

        draw.rectangle(
            (
                left,
                top,
                right,
                bottom,
            ),
            outline="red",
            width=3,
        )

        draw.text(
            (
                left,
                top,
            ),
            str(i),
            fill="red",
            font=find_font(24),
        )

    # --------------------------------------------------------
    # Titles
    # --------------------------------------------------------

    draw.text(
        (
            page_x,
            margin,
        ),
        "Χειρόγραφο κείμενο",
        fill="black",
        font=title_font,
    )

    text_x = (
        margin
        + page_width
        + margin
    )

    draw.text(
        (
            text_x,
            margin,
        ),
        "Κείμενο τεχνητής νοημοσύνης",
        fill="black",
        font=title_font,
    )

    # --------------------------------------------------------
    # Transcription
    # --------------------------------------------------------

    text_y = (
        margin
        + title_height
        + 10
    )

    for prediction in predictions:

        text = prediction["prediction"]

        max_chars = 45

        if len(text) <= max_chars:

            draw.text(
                (
                    text_x,
                    text_y,
                ),
                text,
                fill="black",
                font=text_font,
            )

            text_y += line_height

        else:

            words = text.split()

            current = ""

            for word in words:

                test = (
                    current
                    + " "
                    + word
                ).strip()

                if len(test) > max_chars:

                    draw.text(
                        (
                            text_x,
                            text_y,
                        ),
                        current,
                        fill="black",
                        font=text_font,
                    )

                    text_y += line_height

                    current = word

                else:

                    current = test

            if current:

                draw.text(
                    (
                        text_x,
                        text_y,
                    ),
                    current,
                    fill="black",
                    font=text_font,
                )

                text_y += line_height

            text_y += 10

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    canvas.save(
        output_path,
        quality=95,
    )

    print()
    print("=" * 70)
    print("RESULT SAVED")
    print("=" * 70)

    print(output_path)


# ============================================================
# MAIN
# ============================================================

def main():

    image_path = (
        IMAGE_DIR
        / PAGE_NAME
    )

    xml_path = (
        XML_DIR
        / (
            Path(PAGE_NAME).stem
            + ".xml"
        )
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"Page image not found:\n"
            f"{image_path}"
        )

    if not xml_path.exists():

        raise FileNotFoundError(
            f"PAGE XML not found:\n"
            f"{xml_path}"
        )

    print("=" * 70)
    print("XF53 PAGE OCR EXAMPLE")
    print("=" * 70)

    print(f"Image: {image_path}")
    print(
        f"BLLA:  {SEGMENTATION_MODEL_PATH}"
    )
    print(
        f"OCR:   {MODEL_PATH}"
    )

    # --------------------------------------------------------
    # Load page
    # --------------------------------------------------------

    page_image = Image.open(
        image_path
    ).convert("RGB")

    print(
        f"Page size: "
        f"{page_image.width} x "
        f"{page_image.height}"
    )

    # --------------------------------------------------------
    # Load BLLA
    # --------------------------------------------------------

    segmentation_model = (
        load_segmentation_model()
    )

    # --------------------------------------------------------
    # Detect lines automatically
    # --------------------------------------------------------

    detected_lines = detect_lines(
        page_image,
        segmentation_model,
    )

    # --------------------------------------------------------
    # Load recognition model
    # --------------------------------------------------------

    recognition_model = (
        load_recognition_model()
    )

    # --------------------------------------------------------
    # OCR detected lines
    # --------------------------------------------------------

    predictions = recognize_page(
        recognition_model,
        page_image,
        detected_lines,
    )

    # --------------------------------------------------------
    # Complete transcription
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("COMPLETE AI TRANSCRIPTION")
    print("=" * 70)

    for item in predictions:

        print(
            item["prediction"]
        )

    # --------------------------------------------------------
    # Create visualization
    # --------------------------------------------------------

    create_result_image(
        page_image,
        detected_lines,
        predictions,
        OUTPUT_IMAGE,
    )


if __name__ == "__main__":
    main()