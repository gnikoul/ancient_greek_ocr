from pathlib import Path
import unicodedata

LABELS_FILE = Path("datasets/eparchos/training/labels.txt")

nfc = 0
nfd = 0
neither = 0

with open(LABELS_FILE, encoding="utf-8") as f:
    for line in f:
        _, text = line.rstrip("\n").split("\t", 1)

        if unicodedata.is_normalized("NFC", text):
            nfc += 1

        if unicodedata.is_normalized("NFD", text):
            nfd += 1

        if not (
            unicodedata.is_normalized("NFC", text)
            or unicodedata.is_normalized("NFD", text)
        ):
            neither += 1

print(f"Total:   {nfc + nfd + neither}")
print(f"NFC:     {nfc}")
print(f"NFD:     {nfd}")
print(f"Neither: {neither}")