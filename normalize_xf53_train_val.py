from pathlib import Path

BASE = Path("datasets/XF53")

SETS = {
    "train": BASE / "train",
    "val": BASE / "val",
}

def main():
    total_files = 0
    total_replacements = 0

    for split_name, input_dir in SETS.items():
        output_dir = BASE / f"{split_name}_normalized"
        output_dir.mkdir(exist_ok=True)

        files = list(input_dir.glob("*.gt.txt"))
        replacements = 0

        for gt_file in files:
            text = gt_file.read_text(encoding="utf-8")

            marker_count = text.count("<++>") + text.count("<+>")

            normalized = (
                text
                .replace("<++>", "καὶ")
                .replace("<+>", "καὶ")
            )

            output_file = output_dir / gt_file.name
            output_file.write_text(normalized, encoding="utf-8")

            replacements += marker_count

        total_files += len(files)
        total_replacements += replacements

        print(f"{split_name}: normalized {len(files)} transcripts")
        print(f"{split_name}: replaced {replacements} manuscript markers")
        print(f"Output directory: {output_dir}")

    print()
    print(f"Total transcripts: {total_files}")
    print(f"Total replacements: {total_replacements}")


if __name__ == "__main__":
    main()