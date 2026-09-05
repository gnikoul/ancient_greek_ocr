import shutil
import unicodedata
from pathlib import Path


SOURCE_DIR = Path("datasets/eparchos/training")
OUTPUT_DIR = Path("datasets/eparchos/training_nfd")

SPLITS = ["train", "val", "test"]


def convert_split(split: str) -> None:
    source = SOURCE_DIR / split
    output = OUTPUT_DIR / split

    output.mkdir(parents=True, exist_ok=True)

    image_count = 0
    label_count = 0

    for image_path in source.glob("*.png"):
        shutil.copy2(image_path, output / image_path.name)
        image_count += 1

    for label_path in source.glob("*.gt.txt"):
        text = label_path.read_text(encoding="utf-8")

        # Convert Unicode from NFC/precomposed form to
        # NFD/decomposed form expected by the pretrained model.
        text = unicodedata.normalize("NFD", text)

        output_label = output / label_path.name
        output_label.write_text(text, encoding="utf-8")

        label_count += 1

    print(
        f"{split}: "
        f"{image_count} images, "
        f"{label_count} labels"
    )


def main() -> None:
    print("Preparing NFD dataset")
    print(f"Source: {SOURCE_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"Source directory does not exist: {SOURCE_DIR}"
        )

    if OUTPUT_DIR.exists():
        print(f"WARNING: {OUTPUT_DIR} already exists.")
        print("Removing existing output directory...")
        shutil.rmtree(OUTPUT_DIR)

    for split in SPLITS:
        convert_split(split)

    print()
    print("Done.")
    print(f"NFD dataset created at: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()