from pathlib import Path

DATASET_DIR = Path("datasets/eparchos/training")
IMAGES_DIR = DATASET_DIR / "images"

for split in ["train", "val", "test"]:
    labels_file = DATASET_DIR / f"{split}.txt"
    split_dir = DATASET_DIR / split
    split_dir.mkdir(exist_ok=True)

    count = 0

    with labels_file.open(encoding="utf-8") as f:
        for line in f:
            filename, text = line.rstrip("\n").split("\t", 1)

            src_image = IMAGES_DIR / filename

            if not src_image.exists():
                raise FileNotFoundError(src_image)

            # Keep images in their own split directory
            dst_image = split_dir / filename

            if not dst_image.exists():
                dst_image.symlink_to(src_image.resolve())

            # Kraken path format:
            # image.png -> image.gt.txt
            gt_file = split_dir / f"{src_image.stem}.gt.txt"

            gt_file.write_text(text, encoding="utf-8")

            count += 1

    print(f"{split}: {count} samples")


# Create manifests.
for split in ["train", "val", "test"]:
    split_dir = DATASET_DIR / split
    manifest = DATASET_DIR / f"{split}.lst"

    images = sorted(split_dir.glob("*.png"))

    with manifest.open("w", encoding="utf-8") as f:
        for image in images:
            f.write(f"{image.resolve()}\n")

    print(f"{manifest}: {len(images)} images")