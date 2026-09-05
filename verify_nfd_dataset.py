import unicodedata
from pathlib import Path

from kraken.lib.dataset import GroundTruthDataset
from kraken.models import load_models


MODEL = (
    "/home/gnikoul/.local/share/htrmopo/"
    "a328a8d1-29fb-519a-bf81-3f56a512f080/"
    "ppocr_v6_tau090.safetensors"
)

DATA_DIR = Path("datasets/eparchos/training_nfd")


def main() -> None:
    print("Loading pretrained model...")
    model = load_models(MODEL)[0]
    codec = model.codec

    print(f"Codec size: {len(codec.c2l)}")

    for split in ["train", "val", "test"]:
        directory = DATA_DIR / split
        labels = list(directory.glob("*.gt.txt"))

        print(f"\n{split}:")
        print(f"  Labels: {len(labels)}")

        missing = set()
        non_nfd = 0

        for path in labels:
            text = path.read_text(encoding="utf-8").rstrip("\r\n")

            if unicodedata.normalize("NFD", text) != text:
                non_nfd += 1

            missing.update(
                ch for ch in text
                if ch not in codec.c2l
            )

        print(f"  Non-NFD labels: {non_nfd}")
        print(f"  Characters missing from codec: {len(missing)}")

        if missing:
            for ch in sorted(missing):
                print(
                    f"    {repr(ch)} "
                    f"U+{ord(ch):04X} "
                    f"{unicodedata.name(ch, 'UNKNOWN')}"
                )

    # Inspect one actual label through Kraken's parser.
    sample = sorted((DATA_DIR / "train").glob("*.gt.txt"))[0]

    print("\nSample label:")
    print(f"  File: {sample.name}")

    text = sample.read_text(encoding="utf-8").rstrip("\r\n")
    print(f"  Text: {text}")
    print(f"  NFC:  {unicodedata.normalize('NFC', text)}")
    print(f"  Is NFD: {unicodedata.normalize('NFD', text) == text}")

    print("\nVerification complete.")


if __name__ == "__main__":
    main()