from pathlib import Path
import random


DATASETS = {
    "eparchos": 120,
    "xf53": 54,
    "xf114": 44,
    "xf79": 40,
}

SEED = 42
OUT = Path("datasets/segmentation")


def main():
    rng = random.Random(SEED)

    all_train = []
    all_val = []
    all_test = []

    print("=" * 70)
    print("Creating page-level segmentation splits")
    print("=" * 70)

    for dataset in DATASETS:
        dataset_dir = OUT / dataset
        pages = sorted(dataset_dir.glob("*.xml"))

        if not pages:
            raise RuntimeError(f"No XML files found for {dataset}")

        rng.shuffle(pages)

        n = len(pages)
        n_test = round(n * 0.10)
        n_val = round(n * 0.10)

        test = pages[:n_test]
        val = pages[n_test:n_test + n_val]
        train = pages[n_test + n_val:]

        print(f"\n{dataset.upper()}")
        print("-" * 70)
        print(f"Pages: {n}")
        print(f"Train: {len(train)}")
        print(f"Val:   {len(val)}")
        print(f"Test:  {len(test)}")

        (dataset_dir / "train.lst").write_text(
            "\n".join(str(p) for p in sorted(train)) + "\n"
        )
        (dataset_dir / "val.lst").write_text(
            "\n".join(str(p) for p in sorted(val)) + "\n"
        )
        (dataset_dir / "test.lst").write_text(
            "\n".join(str(p) for p in sorted(test)) + "\n"
        )

        all_train.extend(train)
        all_val.extend(val)
        all_test.extend(test)

    Path(OUT / "train.lst").write_text(
        "\n".join(str(p) for p in sorted(all_train)) + "\n"
    )
    Path(OUT / "val.lst").write_text(
        "\n".join(str(p) for p in sorted(all_val)) + "\n"
    )
    Path(OUT / "test.lst").write_text(
        "\n".join(str(p) for p in sorted(all_test)) + "\n"
    )

    print("\n" + "=" * 70)
    print("TOTAL")
    print("=" * 70)
    print(f"Train: {len(all_train)} pages")
    print(f"Val:   {len(all_val)} pages")
    print(f"Test:  {len(all_test)} pages")


if __name__ == "__main__":
    main()
