from pathlib import Path
import subprocess

BASE_MODEL = Path(
    "/home/gnikoul/.local/share/htrmopo/"
    "a328a8d1-29fb-519a-bf81-3f56a512f080/"
    "ppocr_v6_tau090.safetensors"
)

TRAIN_DATA = Path("datasets/combined_train.lst")
VAL_DATA = Path("datasets/combined_val.lst")

OUTPUT_DIR = Path("models/combined_ppocrv6")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    command = [
        "ketos",
        "train",
        "--load", str(BASE_MODEL),
        "--arch", "ppocrv6",
        "--training-data", str(TRAIN_DATA),
        "--evaluation-data", str(VAL_DATA),
        "--format-type", "path",
        "--resize", "new",
        "--batch-size", "4",
        "--lrate", "0.0001",
        "--optimizer", "AdamW",
        "--schedule", "cosine", # reduceonplateau, step, 1cycle
        "--epochs", "10",
        "--augment",
        "--output", str(OUTPUT_DIR),
    ]

    print("=" * 60)
    print("Combined EPARCHOS + XF53 PP-OCRv6 fine-tuning")
    print("=" * 60)
    print("Command:")
    print(" ".join(command))
    print("=" * 60)

    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()