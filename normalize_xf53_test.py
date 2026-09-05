from pathlib import Path


BASE = Path("datasets/XF53")
INPUT_DIR = BASE / "test"
OUTPUT_DIR = BASE / "test_normalized"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    count = 0
    replacements = 0

    for gt_file in INPUT_DIR.glob("*.gt.txt"):
        text = gt_file.read_text(encoding="utf-8")

        marker_count = text.count("<++>") + text.count("<+>")

        normalized = (
            text
            .replace("<++>", "καὶ")
            .replace("<+>", "καὶ")
        )

        output_file = OUTPUT_DIR / gt_file.name
        output_file.write_text(normalized, encoding="utf-8")

        count += 1
        replacements += marker_count

    print(f"Normalized {count} transcripts")
    print(f"Replaced {replacements} manuscript markers")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()