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

TEST_DIR = Path("datasets/eparchos/training_nfd/test")

CHECKPOINT = "models/eparchos_ppocrv6_lora_nfd/lora-nfd-epoch=08-val_accuracy=0.7185.ckpt"
CHECKPOINT = "models/eparchos_ppocrv6_lora_nfd/lora-nfd-epoch=18-val_accuracy=0.7649.ckpt"
CHECKPOINT = "models/eparchos_ppocrv6_lora_nfd/lora-nfd-epoch=09-val_accuracy=0.7218.ckpt"


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
    return unicodedata.normalize("NFD", text.strip())


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


def load_lora_model(checkpoint_path):

    print("\n" + "=" * 70)
    print("Loading:", checkpoint_path)
    print("=" * 70)

    model = load_models(MODEL)[0]

    print("Pretrained codec size:", len(model.codec))

    lora_config = LoraConfig(
        r=32,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=LORA_TARGETS,
        bias="none",
    )

    model.nn = get_peft_model(
        model.nn,
        lora_config,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = checkpoint["state_dict"]

    lora_state = {}

    for key, value in state_dict.items():

        if "lora_" not in key:
            continue

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
        for key in missing_lora:
            print(" ", key)
        raise RuntimeError("Some LoRA weights were not loaded.")

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
    print("NFD LORA ERROR ANALYSIS")
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

    total_ref_spaces = 0
    total_pred_spaces = 0

    # Store examples:
    # (character_error_rate, filename, reference, prediction, word_errors)
    examples = []

    image_paths = sorted(TEST_DIR.glob("*.png"))

    print(f"Test samples: {len(image_paths)}")
    print("Running recognition...\n")

    for i, image_path in enumerate(image_paths, 1):

        label_path = image_path.with_suffix(".gt.txt")

        reference = label_path.read_text(
            encoding="utf-8"
        )

        reference = normalize(reference)

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

        # Character errors
        char_errors = edit_distance(
            reference,
            prediction,
        )

        char_count = len(reference)

        # Word errors
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

        # Space statistics
        ref_spaces = reference.count(" ")
        pred_spaces = prediction.count(" ")

        total_ref_spaces += ref_spaces
        total_pred_spaces += pred_spaces

        cer = char_errors / char_count
        wer = word_errors / word_count

        # Keep interesting examples.
        #
        # We want cases where CER is relatively low but WER is high.
        if cer < 0.20 and wer > 0.40:
            examples.append(
                (
                    cer,
                    filename := image_path.name,
                    reference,
                    prediction,
                    word_errors,
                    wer,
                )
            )

        if i % 25 == 0:
            print(f"Processed {i}/{len(image_paths)}")

    cer = total_char_errors / total_chars
    wer = total_word_errors / total_words

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)

    print(f"Characters:       {total_chars}")
    print(f"Character errors: {total_char_errors}")
    print(f"CER:              {cer:.4f} ({cer * 100:.2f}%)")

    print()

    print(f"Words:            {total_words}")
    print(f"Word errors:      {total_word_errors}")
    print(f"WER:              {wer:.4f} ({wer * 100:.2f}%)")

    print()
    print("=" * 60)
    print("SPACE ANALYSIS")
    print("=" * 60)

    print(f"Reference spaces:   {total_ref_spaces}")
    print(f"Predicted spaces:   {total_pred_spaces}")
    print(
        f"Difference:         "
        f"{total_pred_spaces - total_ref_spaces:+d}"
    )

    print(
        f"Relative difference: "
        f"{(total_pred_spaces - total_ref_spaces) / total_ref_spaces * 100:+.2f}%"
    )

    # --------------------------------------------------
    # Print examples
    # --------------------------------------------------

    examples.sort(key=lambda x: x[0])

    print("\n" + "=" * 60)
    print("LOW CER / HIGH WER EXAMPLES")
    print("=" * 60)

    if not examples:
        print("No examples matched the selected criteria.")

    else:

        for (
            example_cer,
            filename,
            reference,
            prediction,
            word_errors,
            example_wer,
        ) in examples[:15]:

            print()
            print(f"File: {filename}")
            print(
                f"CER={example_cer:.3f} | "
                f"WER={example_wer:.3f} | "
                f"word errors={word_errors}"
            )
            print("REF : ", reference)
            print("PRED: ", prediction)

    return cer, wer


def main():

    model = load_lora_model(CHECKPOINT)

    evaluate(model)

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()