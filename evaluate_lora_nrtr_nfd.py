from peft import LoraConfig, get_peft_model
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
LORA_CHECKPOINT = "models/eparchos_ppocrv6_lora_nrtr_nfd/lora-nrtr-nfd-epoch=05-val_accuracy=0.7144.ckpt"
TEST_DIR = Path("datasets/eparchos/training_nfd/test")


def normalize(text):
    return unicodedata.normalize("NFD", text.strip())


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


def load_model():

    print("=" * 70)
    print("LOADING BASE MODEL + LORA")
    print("=" * 70)

    model = load_models(MODEL)[0]

    print("Codec size:", len(model.codec))

    # Same LoRA configuration used during training.
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "neck.svtr_block.0.mixer.qkv",
            "neck.svtr_block.0.mixer.proj",
            "neck.svtr_block.0.mlp.fc1",
            "neck.svtr_block.0.mlp.fc2",
            "neck.svtr_block.1.mixer.qkv",
            "neck.svtr_block.1.mixer.proj",
            "neck.svtr_block.1.mlp.fc1",
            "neck.svtr_block.1.mlp.fc2",
        ],
    )

    model.nn = get_peft_model(model.nn, lora_config)

    # Load Lightning checkpoint.
    checkpoint = torch.load(
        LORA_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = checkpoint["state_dict"]

    # Keep only LoRA weights.
    lora_state = {
        key.replace("net.nn.", "", 1): value
        for key, value in state_dict.items()
        if "lora_" in key
    }

    print("LoRA tensors in checkpoint:", len(lora_state))

    missing, unexpected = model.nn.load_state_dict(
        lora_state,
        strict=False,
    )

    missing_lora = [
        key for key in missing
        if "lora_" in key
    ]

    unexpected_lora = [
        key for key in unexpected
        if "lora_" in key
    ]

    print("Missing LoRA tensors:", len(missing_lora))
    print("Unexpected LoRA tensors:", len(unexpected_lora))

    if missing_lora or unexpected_lora:
        raise RuntimeError(
            f"LoRA loading failed.\n"
            f"Missing: {missing_lora}\n"
            f"Unexpected: {unexpected_lora}"
        )

    # PEFT needs the Kraken input information before inference preparation.
    model.nn.input = model.input

    config = RecognitionInferenceConfig(
        accelerator="cuda",
        devices=1,
        batch_size=1,
        padding=0,
    )

    model.prepare_for_inference(config)

    print("Device:", next(model.parameters()).device)

    return model


def evaluate(model):

    print("\n" + "=" * 70)
    print("EVALUATING LORA + NRTR ON NFD TEST SET")
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

    image_paths = sorted(TEST_DIR.glob("*.png"))

    print(f"Test samples: {len(image_paths)}")
    print("Running recognition...\n")

    for i, image_path in enumerate(image_paths, 1):

        label_path = image_path.with_suffix(".gt.txt")

        if not label_path.exists():
            raise FileNotFoundError(
                f"Missing label: {label_path}"
            )

        # NFD reference
        reference = label_path.read_text(
            encoding="utf-8"
        )
        reference = normalize(reference)

        image = Image.open(image_path)

        # Same preprocessing as all previous evaluations.
        with model._fabric.init_tensor():
            ts_im = transforms(image)

        seqs = ts_im.unsqueeze(0)

        seq_lens = torch.tensor(
            [ts_im.shape[2]],
            device=seqs.device,
        )

        # Same recognition path as before.
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

        if i % 25 == 0:
            print(f"Processed {i}/{len(image_paths)}")

    cer = total_char_errors / total_chars
    wer = total_word_errors / total_words

    print("\n" + "=" * 60)
    print("BASELINE NFD RESULTS")
    print("=" * 60)

    print(f"Characters:       {total_chars}")
    print(f"Character errors: {total_char_errors}")
    print(f"CER:              {cer:.4f} ({cer * 100:.2f}%)")
    print()

    print(f"Words:            {total_words}")
    print(f"Word errors:      {total_word_errors}")
    print(f"WER:              {wer:.4f} ({wer * 100:.2f}%)")

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    print(
        f"{'Model':<25}"
        f"{'CER':>12}"
        f"{'WER':>12}"
    )

    print("-" * 50)

    print(
        f"{'NFD LoRA + NRTR':<25}"
        f"{cer * 100:>11.2f}%"
        f"{wer * 100:>11.2f}%"
    )

    print(
        f"{'NFD LoRA':<25}"
        f"{17.94:>11.2f}%"
        f"{70.28:>11.2f}%"
    )


def main():

    model = load_model()

    evaluate(model)

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()