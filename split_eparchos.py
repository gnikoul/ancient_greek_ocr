from pathlib import Path
import random

DATASET_DIR = Path("datasets/eparchos/training")
LABELS_FILE = DATASET_DIR / "labels.txt"

random.seed(42)

samples = []

with LABELS_FILE.open(encoding="utf-8") as f:
    for line in f:
        filename, text = line.rstrip("\n").split("\t", 1)
        samples.append((filename, text))

random.shuffle(samples)

n = len(samples)
train_end = int(0.8 * n)
val_end = train_end + int(0.1 * n)

train = samples[:train_end]
val = samples[train_end:val_end]
test = samples[val_end:]

for name, data in [
    ("train.txt", train),
    ("val.txt", val),
    ("test.txt", test),
]:
    with (DATASET_DIR / name).open("w", encoding="utf-8") as f:
        for filename, text in data:
            f.write(f"{filename}\t{text}\n")

print(f"Total:      {n}")
print(f"Train:      {len(train)}")
print(f"Validation: {len(val)}")
print(f"Test:       {len(test)}")