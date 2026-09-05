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


BASE_MODEL = (
    "/home/gnikoul/.local/share/htrmopo/"
    "a328a8d1-29fb-519a-bf81-3f56a512f080/"
    "ppocr_v6_tau090.safetensors"
)

DATASET = Path("datasets/eparchos/training")
IMAGE_DIR = DATASET / "images"
TEST_FILE = DATASET / "test.txt"


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


CHECKPOINTS = {
    "lora_nrtr": Path(
        "models/eparchos_ppocrv6_lora/"
        "lora-epoch=09-val_accuracy=0.8010.ckpt"
    ),
    "lora_only": Path(
        "models/eparchos_ppocrv6_lora/"
        "lora-epoch=08-val_accuracy=0.6437.ckpt"
    ),
    "lora_only_cosine": Path(
        "models/eparchos_ppocrv6_lora_cosine/"
        "lora-epoch=09-val_accuracy=0.5939.ckpt"
    ),
}


def normalize(text):
    return unicodedata.normalize("NFC", text.strip())


def make_lora_model():
    model = load_models(BASE_MODEL)[0]

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=LORA_TARGETS,
    )

    model.nn = get_peft_model(model.nn, config)

    # PEFT wrapper hides Kraken's input attribute.
    model.nn.input = model.input

    return model


def load_lora_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    state_dict = checkpoint["state_dict"]

    lora_state = {}

    for key, value in state_dict.items():
        if "lora_" not in key:
            continue

        # Lightning checkpoint:
        # net.nn.base_model.model....
        #
        # PEFT model:
        # base_model.model....
        if key.startswith("net.nn."):
            key = key[len("net.nn."):]

        lora_state[key] = value

    result = model.nn.load_state_dict(
        lora_state,
        strict=False,
    )

    missing_lora = [
        x for x in result.missing_keys
        if "lora_" in x
    ]

    unexpected_lora = [
        x for x in result.unexpected_keys
        if "lora_" in x
    ]

    print(f"  LoRA tensors loaded: {len(lora_state)}")
    print(f"  Missing LoRA keys: {len(missing_lora)}")
    print(f"  Unexpected LoRA keys: {len(unexpected_lora)}")

    if missing_lora:
        print("  Missing:")
        for x in missing_lora:
            print(f"    {x}")

    if unexpected_lora:
        print("  Unexpected:")
        for x in unexpected_lora:
            print(f"    {x}")


def prepare_model(model):
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

    return transforms


def predict(model, transforms, image_path):
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

    return normalize(prediction)


def load_samples():
    with open(TEST_FILE, encoding="utf-8") as f:
        samples = [
            line.rstrip("\n").split("\t", 1)
            for line in f
        ]

    return samples[:10]


def main():

    samples = load_samples()

    print("Loading baseline model...")
    baseline = load_models(BASE_MODEL)[0]
    baseline_transforms = prepare_model(baseline)

    print("\nLoading LoRA models...")

    lora_models = {}

    for name, checkpoint in CHECKPOINTS.items():
        print(f"\n{name}:")
        print(f"  checkpoint: {checkpoint}")

        model = make_lora_model()
        load_lora_checkpoint(model, checkpoint)

        transforms = prepare_model(model)

        lora_models[name] = (model, transforms)

    print("\n" + "=" * 100)
    print("PREDICTIONS")
    print("=" * 100)

    for i, (filename, reference) in enumerate(samples, 1):

        image_path = IMAGE_DIR / filename

        baseline_prediction = predict(
            baseline,
            baseline_transforms,
            image_path,
        )

        predictions = {}

        for name, (model, transforms) in lora_models.items():
            predictions[name] = predict(
                model,
                transforms,
                image_path,
            )

        print("\n" + "-" * 100)
        print(f"{i}. {filename}")
        print("-" * 100)

        print(f"REF:")
        print(reference)

        print(f"\nBASELINE:")
        print(baseline_prediction)

        for name, prediction in predictions.items():
            print(f"\n{name}:")
            print(prediction)

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()