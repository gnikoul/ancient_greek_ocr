from pathlib import Path
import unicodedata

labels = Path("datasets/eparchos/training/labels.txt")

for i, line in enumerate(labels.open(encoding="utf-8"), 1):
    _, text = line.rstrip("\n").split("\t", 1)

    if not unicodedata.is_normalized("NFC", text):
        print(f"Line {i}: {repr(text)}")
        print(f"NFC: {repr(unicodedata.normalize('NFC', text))}")
        print(f"NFD: {repr(unicodedata.normalize('NFD', text))}")