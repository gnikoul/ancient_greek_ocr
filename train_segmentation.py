from pathlib import Path
import subprocess


BASE_MODEL = Path(
    "/home/gnikoul/miniconda3/envs/grc/lib/python3.11/"
    "site-packages/kraken/blla.mlmodel"
)

TRAIN_DATA = Path("datasets/segmentation/train.lst")
VAL_DATA = Path("datasets/segmentation/val.lst")

OUTPUT_DIR = Path("models/segmentation_finetuned")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    command = [
        "ketos",
        "segtrain",
        "--load", str(BASE_MODEL),
        "--training-data", str(TRAIN_DATA),
        "--evaluation-data", str(VAL_DATA),
        "--format-type", "page",
        "--resize", "new",
        "--lrate", "0.0001",
        "--optimizer", "AdamW",
        "--schedule", "cosine",
        "--epochs", "10",
        "--augment",
        "--output", str(OUTPUT_DIR),
    ]

    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()