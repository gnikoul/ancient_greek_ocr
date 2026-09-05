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
    "models/eparchos_ppocrv6_test/"
    "best_0.5956.safetensors"
)

DATASET = Path("datasets/eparchos/training")
IMAGE_DIR = DATASET / "images"
TEST_FILE = DATASET / "test.txt"


def normalize(text):
    return unicodedata.normalize("NFC", text.strip())


def edit_distance(a, b):
    prev = list(range(len(b) + 1))

    for i, ca in enumerate(a, 1):
        curr = [i]

        for j, cb in enumerate(b, 1):
            curr.append(min(
                curr[-1] + 1,
                prev[j] + 1,
                prev[j - 1] + (ca != cb)
            ))

        prev = curr

    return prev[-1]


print("Loading fine-tuned model...")

model = load_models(MODEL)[0]

config = RecognitionInferenceConfig(
    accelerator="cuda",
    devices=1,
    batch_size=1,
    padding=0,
)

model.prepare_for_inference(config)

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


with open(TEST_FILE, encoding="utf-8") as f:
    test_samples = [
        line.rstrip("\n").split("\t", 1)
        for line in f
    ]


total_char_errors = 0
total_chars = 0

total_word_errors = 0
total_words = 0


print(f"Evaluating {len(test_samples)} test samples...")
print("=" * 80)


for i, (filename, reference) in enumerate(test_samples, 1):

    reference = normalize(reference)

    image_path = IMAGE_DIR / filename
    image = Image.open(image_path)

    with model._fabric.init_tensor():
        ts_im = transforms(image)

    seqs = ts_im.unsqueeze(0)

    seq_lens = torch.tensor(
        [ts_im.shape[2]],
        device=seqs.device,
    )

    preds, olens = model._rec_predict(
        seqs,
        seq_lens,
    )

    prediction = "".join(
        x[0] for x in preds[0]
    )

    prediction = normalize(prediction)

    char_errors = edit_distance(
        reference,
        prediction,
    )

    reference_words = reference.split()
    prediction_words = prediction.split()

    word_errors = edit_distance(
        reference_words,
        prediction_words,
    )

    total_char_errors += char_errors
    total_chars += len(reference)

    total_word_errors += word_errors
    total_words += len(reference_words)

    if i % 25 == 0 or i == len(test_samples):
        print(
            f"{i:4d}/{len(test_samples)}  "
            f"CER: {total_char_errors / total_chars:.4%}  "
            f"WER: {total_word_errors / total_words:.4%}"
        )


cer = total_char_errors / total_chars
wer = total_word_errors / total_words

print()
print("=" * 80)
print("RESULT")
print("=" * 80)
print(f"Characters:     {total_chars}")
print(f"Char errors:    {total_char_errors}")
print(f"CER:            {cer:.2%}")
print()
print(f"Words:          {total_words}")
print(f"Word errors:    {total_word_errors}")
print(f"WER:            {wer:.2%}")