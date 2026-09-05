import torch
import unicodedata
from pathlib import Path
from PIL import Image

from kraken_monkey_patch import apply_kraken_monkey_patch

apply_kraken_monkey_patch()

from kraken.models import load_models
from kraken.configs import RecognitionInferenceConfig
from kraken.lib.dataset import ImageInputTransforms


MODEL = (
    "/home/gnikoul/.local/share/htrmopo/"
    "a328a8d1-29fb-519a-bf81-3f56a512f080/"
    "ppocr_v6_tau090.safetensors"
)

DATASET = Path("datasets/eparchos/training")
IMAGE_DIR = DATASET / "images"
TEST_FILE = DATASET / "test.txt"
RESULTS_FILE = DATASET / "baseline_results.tsv"


def normalize(text):
    return unicodedata.normalize("NFC", text.strip())


def edit_distance(a, b):
    """Levenshtein distance."""
    prev = list(range(len(b) + 1))

    for i, ca in enumerate(a, 1):
        curr = [i]

        for j, cb in enumerate(b, 1):
            curr.append(min(
                curr[-1] + 1,       # insertion
                prev[j] + 1,        # deletion
                prev[j - 1] + (ca != cb)  # substitution
            ))

        prev = curr

    return prev[-1]


# --------------------------------------------------
# Load model
# --------------------------------------------------

print("Loading model...")

model = load_models(MODEL)[0]

config = RecognitionInferenceConfig(
    accelerator="cuda",
    devices=1,
    batch_size=1,
    padding=0,
)

model.prepare_for_inference(config)

print("Model:", type(model).__name__)
print("Device:", next(model.parameters()).device)


# --------------------------------------------------
# Create preprocessing pipeline
# --------------------------------------------------

batch, channels, height, width = model.input

transforms = ImageInputTransforms(
    batch,
    height,
    width,
    channels,
    (model._inf_config.padding, 0),
    False,
    dtype=model._m_dtype,
)


# --------------------------------------------------
# Evaluate test set
# --------------------------------------------------

total_char_errors = 0
total_chars = 0

total_word_errors = 0
total_words = 0

results = []

with open(TEST_FILE, encoding="utf-8") as f:
    test_samples = [line.rstrip("\n").split("\t", 1) for line in f]

print(f"Test samples: {len(test_samples)}")
print("Running recognition...\n")


for i, (filename, reference) in enumerate(test_samples, 1):

    reference = normalize(reference)

    image_path = IMAGE_DIR / filename
    image = Image.open(image_path)

    # Preprocess image
    with model._fabric.init_tensor():
        ts_im = transforms(image)

    seqs = ts_im.unsqueeze(0)

    # IMPORTANT: lengths must be on the same device as the model
    seq_lens = torch.tensor(
        [ts_im.shape[2]],
        device=seqs.device
    )

    # Recognition
    preds, olens = model._rec_predict(seqs, seq_lens)
    

    prediction = "".join(x[0] for x in preds[0])
    prediction = normalize(prediction)

    # CER
    char_errors = edit_distance(reference, prediction)
    char_count = len(reference)

    # WER
    ref_words = reference.split()
    pred_words = prediction.split()

    word_errors = edit_distance(ref_words, pred_words)
    word_count = len(ref_words)

    total_char_errors += char_errors
    total_chars += char_count

    total_word_errors += word_errors
    total_words += word_count

    results.append(
        (filename, reference, prediction, char_errors, word_errors)
    )

    print(
        f"[{i:3d}/{len(test_samples)}] "
        f"{filename} | "
        f"CER={char_errors / char_count:.3f} | "
        f"WER={word_errors / word_count:.3f}"
    )


# --------------------------------------------------
# Final scores
# --------------------------------------------------

cer = total_char_errors / total_chars
wer = total_word_errors / total_words

print("\n" + "=" * 50)
print("BASELINE RESULTS")
print("=" * 50)
print(f"Characters: {total_chars}")
print(f"Character errors: {total_char_errors}")
print(f"CER: {cer:.4f} ({cer * 100:.2f}%)")
print()
print(f"Words: {total_words}")
print(f"Word errors: {total_word_errors}")
print(f"WER: {wer:.4f} ({wer * 100:.2f}%)")
print("=" * 50)


# --------------------------------------------------
# Save detailed results
# --------------------------------------------------

with open(RESULTS_FILE, "w", encoding="utf-8") as f:

    f.write("filename\treference\tprediction\tchar_errors\tword_errors\n")

    for filename, reference, prediction, char_errors, word_errors in results:
        f.write(
            f"{filename}\t"
            f"{reference}\t"
            f"{prediction}\t"
            f"{char_errors}\t"
            f"{word_errors}\n"
        )

print(f"\nDetailed results saved to: {RESULTS_FILE}")