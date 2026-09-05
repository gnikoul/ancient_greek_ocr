import torch
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from kraken_monkey_patch import apply_kraken_monkey_patch

apply_kraken_monkey_patch()

from kraken.models import load_models
from kraken.configs import RecognitionInferenceConfig
from kraken.lib.dataset import ImageInputTransforms


# ============================================================
# CONFIGURATION
# ============================================================

# Change this to any page you want to show.
PAGE_NAME = "15_01_0053_0006_f_3r_res.png"

IMAGE_DIR = Path("datasets/XF53/images")
XML_DIR = Path("datasets/XF53/xml")

MODEL_PATH = Path(
    "models/combined_ppocrv6/"
    "best_0.8614.safetensors"
)

OUTPUT_IMAGE = Path("ocr_example.png")


# ============================================================
# MODEL
# ============================================================

def load_model():

    print("=" * 70)
    print("LOADING EPARCHOS+XF53 MODEL")
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

    print("Device:", next(model.parameters()).device)

    return model


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(text):
    return unicodedata.normalize("NFD", text.strip())


# ============================================================
# XML LINE EXTRACTION
# ============================================================

def get_line_boxes(xml_path):

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # PAGE XML uses a namespace.
    namespace = {
        "page": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
    }

    lines = []

    for text_line in root.findall(".//page:TextLine", namespace):

        coords = text_line.find("page:Coords", namespace)

        if coords is None:
            continue

        points_string = coords.attrib.get("points")

        if not points_string:
            continue

        points = []

        for point in points_string.split():

            x, y = point.split(",")

            points.append(
                (int(x), int(y))
            )

        if not points:
            continue

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        left = min(xs)
        top = min(ys)
        right = max(xs)
        bottom = max(ys)

        lines.append({
            "id": text_line.attrib.get("id", ""),
            "bbox": (left, top, right, bottom),
            "top": top,
        })

    # Important:
    # XML order is normally already correct, but sorting by
    # vertical position makes the reconstruction robust.
    lines.sort(key=lambda x: x["top"])

    return lines


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
        (model._inf_config.padding, 0),
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
        x[0] for x in preds[0]
    )

    return normalize(prediction)


# ============================================================
# RECOGNIZE PAGE
# ============================================================

def recognize_page(model, page_image, lines):

    predictions = []

    print()
    print("=" * 70)
    print("RECOGNIZING PAGE")
    print("=" * 70)

    print(f"Lines detected: {len(lines)}")
    print()

    for i, line in enumerate(lines, 1):

        left, top, right, bottom = line["bbox"]

        # Small horizontal/vertical safety padding.
        # The XML coordinates are already line-level bounding boxes,
        # so keep this small.
        padding_x = 2
        padding_y = 2

        left = max(0, left - padding_x)
        top = max(0, top - padding_y)

        right = min(page_image.width, right + padding_x)
        bottom = min(page_image.height, bottom + padding_y)

        crop = page_image.crop(
            (left, top, right, bottom)
        )

        prediction = predict_line(
            model,
            crop,
        )

        predictions.append({
            "id": line["id"],
            "bbox": line["bbox"],
            "prediction": prediction,
        })

        print(
            f"{i:3d}/{len(lines):3d} "
            f"{line['id']:<10} "
            f"{prediction}"
        )

    return predictions


# ============================================================
# CREATE DISPLAY IMAGE
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


def create_result_image(
    original_page,
    predictions,
    output_path,
):

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    page_width = 1000

    scale = page_width / original_page.width

    page_height = int(
        original_page.height * scale
    )

    page = original_page.resize(
        (page_width, page_height)
    )

    # Right-hand transcription panel.
    text_width = 850

    margin = 40

    title_height = 100

    line_height = 65

    transcription_height = max(
        500,
        title_height
        + len(predictions) * line_height
        + margin * 2,
    )

    total_width = (
        page_width
        + text_width
        + margin * 3
    )

    total_height = max(
        page_height + margin * 2,
        transcription_height,
    )

    canvas = Image.new(
        "RGB",
        (total_width, total_height),
        "white",
    )

    draw = ImageDraw.Draw(canvas)

    # --------------------------------------------------------
    # Original page
    # --------------------------------------------------------

    page_x = margin
    page_y = margin + title_height

    canvas.paste(
        page,
        (page_x, page_y),
    )

    # --------------------------------------------------------
    # Fonts
    # --------------------------------------------------------

    title_font = find_font(38)
    text_font = find_font(30)

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

        # Wrap very long lines.
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
                    current + " " + word
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

    image_path = IMAGE_DIR / PAGE_NAME

    xml_path = XML_DIR / (
        Path(PAGE_NAME).stem + ".xml"
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"Page image not found:\n{image_path}"
        )

    if not xml_path.exists():

        raise FileNotFoundError(
            f"PAGE XML not found:\n{xml_path}"
        )

    print("=" * 70)
    print("XF53 PAGE OCR EXAMPLE")
    print("=" * 70)

    print(f"Image: {image_path}")
    print(f"XML:   {xml_path}")
    print(f"Model: {MODEL_PATH}")

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
    # Read XML
    # --------------------------------------------------------

    lines = get_line_boxes(
        xml_path
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # OCR every line
    # --------------------------------------------------------

    predictions = recognize_page(
        model,
        page_image,
        lines,
    )

    # --------------------------------------------------------
    # Print complete transcription
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("COMPLETE AI TRANSCRIPTION")
    print("=" * 70)

    for item in predictions:
        print(item["prediction"])

    # --------------------------------------------------------
    # Create side-by-side presentation
    # --------------------------------------------------------

    create_result_image(
        page_image,
        predictions,
        OUTPUT_IMAGE,
    )


if __name__ == "__main__":
    main()