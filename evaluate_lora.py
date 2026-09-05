import torch
import unicodedata
from pathlib import Path

from PIL import Image
from peft import LoraConfig, get_peft_model

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


CHECKPOINTS = {
    "lora_nrtr": (
        "models/eparchos_ppocrv6_lora/"
        "lora-epoch=09-val_accuracy=0.8010.ckpt"
    ),
    "lora_only": (
        "models/eparchos_ppocrv6_lora/"
        "lora-epoch=08-val_accuracy=0.6437.ckpt"
    ),
    "lora_only_cosine": (
        "models/eparchos_ppocrv6_lora_cosine/"
        "lora-epoch=09-val_accuracy=0.5939.ckpt"
    ),
}


LORA_TARGETS = [
    "neck.svtr_block.0.mixer.qkv",
    "neck.svtr_block.0.mixer.proj",
    "neck.svtr_block.0.mlp.fc1",
    "neck.svtr_block.0.mlp.fc2",
    "neck.svtr_block.1.mixer.qkv",
    "neck.svtr_block.1.mixer.proj",
    "neck.svtr_block.1.mlp.fc1",
    "neck.svtr_block.1.mlp.fc2",
]


def normalize(text):
    return unicodedata.normalize("NFC", text.strip())


def edit_distance(a, b):
    """Levenshtein distance."""
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


def load_lora_model(checkpoint_path):
    print("\n" + "=" * 70)
    print("Loading:", checkpoint_path)
    print("=" * 70)

    # Load original Kraken model.
    model = load_models(MODEL)[0]

    # Create exactly the same PEFT configuration used during training.
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=LORA_TARGETS,
        bias="none",
    )

    model.nn = get_peft_model(model.nn, lora_config)

    # Load Lightning checkpoint.
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = checkpoint["state_dict"]

    # Extract LoRA weights.
    lora_state = {}

    for key, value in state_dict.items():

        if "lora_" not in key:
            continue

        # Checkpoint:
        #   net.nn.base_model.model....
        #
        # PEFT model:
        #   base_model.model....
        if key.startswith("net.nn."):
            key = key[len("net.nn."):]

        lora_state[key] = value

    print(f"LoRA tensors found: {len(lora_state)}")

    missing, unexpected = model.nn.load_state_dict(
        lora_state,
        strict=False,
    )

    missing_lora = [
        key for key in missing
        if "lora_" in key
    ]

    print(f"Missing LoRA keys: {len(missing_lora)}")
    print(f"Unexpected keys: {len(unexpected)}")

    if missing_lora:
        print("\nMissing LoRA keys:")
        for key in missing_lora:
            print(" ", key)
        raise RuntimeError("Some LoRA weights were not loaded.")

    if unexpected:
        print("\nUnexpected keys:")
        for key in unexpected:
            print(" ", key)

    # --------------------------------------------------
    # Important:
    #
    # Kraken's inference code expects model.nn to expose
    # attributes such as `input`.
    #
    # PEFT wraps the original model, so copy the required
    # Kraken metadata onto the PEFT wrapper.
    # --------------------------------------------------

    model.nn.input = model.input

    # The underlying recognizer has the actual inference
    # methods/metadata. PEFT forwards most attributes, but
    # Kraken directly accesses `input`.

    config = RecognitionInferenceConfig(
        accelerator="cuda",
        devices=1,
        batch_size=1,
        padding=0,
    )

    model.prepare_for_inference(config)

    print("Model:", type(model).__name__)
    print("Device:", next(model.parameters()).device)

    return model


def evaluate(model, name):

    print("\n" + "=" * 70)
    print(f"EVALUATING: {name}")
    print("=" * 70)

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

    total_char_errors = 0
    total_chars = 0

    total_word_errors = 0
    total_words = 0

    with open(TEST_FILE, encoding="utf-8") as f:
        test_samples = [
            line.rstrip("\n").split("\t", 1)
            for line in f
        ]

    print(f"Test samples: {len(test_samples)}")
    print("Running recognition...\n")

    for i, (filename, reference) in enumerate(test_samples, 1):

        reference = normalize(reference)

        image_path = IMAGE_DIR / filename
        image = Image.open(image_path)

        # Same preprocessing as baseline.
        with model._fabric.init_tensor():
            ts_im = transforms(image)

        seqs = ts_im.unsqueeze(0)

        seq_lens = torch.tensor(
            [ts_im.shape[2]],
            device=seqs.device,
        )

        # Same recognition path as baseline.
        preds, olens = model._rec_predict(
            seqs,
            seq_lens,
        )

        prediction = "".join(
            x[0] for x in preds[0]
        )

        prediction = normalize(prediction)

        # CER
        char_errors = edit_distance(
            reference,
            prediction,
        )

        char_count = len(reference)

        # WER
        ref_words = reference.split()
        pred_words = prediction.split()

        word_errors = edit_distance(
            ref_words,
            pred_words,
        )

        word_count = len(ref_words)

        total_char_errors += char_errors
        total_chars += char_count

        total_word_errors += word_errors
        total_words += word_count

        print(
            f"[{i:3d}/{len(test_samples)}] "
            f"{filename} | "
            f"CER={char_errors / char_count:.3f} | "
            f"WER={word_errors / word_count:.3f}"
        )

    cer = total_char_errors / total_chars
    wer = total_word_errors / total_words

    print("\n" + "=" * 50)
    print(f"{name.upper()} RESULTS")
    print("=" * 50)

    print(f"Characters:       {total_chars}")
    print(f"Character errors: {total_char_errors}")
    print(f"CER:              {cer:.4f} ({cer * 100:.2f}%)")
    print()
    print(f"Words:            {total_words}")
    print(f"Word errors:      {total_word_errors}")
    print(f"WER:              {wer:.4f} ({wer * 100:.2f}%)")

    return cer, wer


def main():

    results = {}

    for name, checkpoint in CHECKPOINTS.items():

        model = load_lora_model(checkpoint)

        results[name] = evaluate(
            model,
            name,
        )

        del model
        torch.cuda.empty_cache()

    print("\n\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    print(
        f"{'Model':<25}"
        f"{'CER':>12}"
        f"{'WER':>12}"
    )

    print("-" * 50)

    print(
        f"{'Baseline':<25}"
        f"{19.65:>11.2f}%"
        f"{60.39:>11.2f}%"
    )

    for name, (cer, wer) in results.items():

        print(
            f"{name:<25}"
            f"{cer * 100:>11.2f}%"
            f"{wer * 100:>11.2f}%"
        )


if __name__ == "__main__":
    main()