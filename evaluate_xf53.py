import torch
import unicodedata
from pathlib import Path
from PIL import Image

from kraken_monkey_patch import apply_kraken_monkey_patch

apply_kraken_monkey_patch()

from kraken.models import load_models
from kraken.configs import RecognitionInferenceConfig
from kraken.lib.dataset import ImageInputTransforms


BASE_MODEL = "/home/gnikoul/.local/share/htrmopo/a328a8d1-29fb-519a-bf81-3f56a512f080/ppocr_v6_tau090.safetensors"
FINETUNED_MODEL = "models/eparchos_ppocrv6_nfd/best_0.9637.safetensors"

IMAGE_DIR = Path("datasets/XF53/test")
LABEL_DIR = Path("datasets/XF53/test_normalized")

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


def load_model(model_path, name):

    print("=" * 70)
    print(f"LOADING {name.upper()}")
    print("=" * 70)

    model = load_models(model_path)[0]

    print("Codec size:", len(model.codec))

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

    results = []
    
    print("\n" + "=" * 70)
    print("EVALUATING BASELINE ON NFD TEST SET")
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

    image_paths = sorted(IMAGE_DIR.glob("*.png"))

    print(f"Test samples: {len(image_paths)}")
    print("Running recognition...\n")

    for i, image_path in enumerate(image_paths, 1):

        label_path = LABEL_DIR / f"{image_path.stem}.gt.txt"

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
        results.append({
            "image": image_path.name,
            "reference": reference,
            "prediction": prediction,
        })
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
        f"{'Baseline NFD':<25}"
        f"{cer * 100:>11.2f}%"
        f"{wer * 100:>11.2f}%"
    )
    return results

def compare_predictions(baseline_results, finetuned_results):

    fixed_errors = []
    broken_errors = []

    for base, ft in zip(baseline_results, finetuned_results):

        reference = base["reference"]
        base_prediction = base["prediction"]
        ft_prediction = ft["prediction"]

        base_distance = edit_distance(reference, base_prediction)
        ft_distance = edit_distance(reference, ft_prediction)

        if ft_distance < base_distance:
            fixed_errors.append({
                "image": base["image"],
                "reference": reference,
                "baseline": base_prediction,
                "finetuned": ft_prediction,
                "baseline_errors": base_distance,
                "finetuned_errors": ft_distance,
            })

        elif ft_distance > base_distance:
            broken_errors.append({
                "image": base["image"],
                "reference": reference,
                "baseline": base_prediction,
                "finetuned": ft_prediction,
                "baseline_errors": base_distance,
                "finetuned_errors": ft_distance,
            })

    print("\n" + "=" * 70)
    print("ERROR REDUCTION BY IMAGE")
    print("=" * 70)

    print(f"Images improved:  {len(fixed_errors)}")
    print(f"Images worsened:  {len(broken_errors)}")

    total_fixed = sum(
        x["baseline_errors"] - x["finetuned_errors"]
        for x in fixed_errors
    )

    total_broken = sum(
        x["finetuned_errors"] - x["baseline_errors"]
        for x in broken_errors
    )

    print(f"Errors eliminated: {total_fixed}")
    print(f"Errors introduced:  {total_broken}")
def print_best_improvements(baseline_results, finetuned_results, n=20):

    improvements = []

    for base, ft in zip(baseline_results, finetuned_results):

        reference = base["reference"]
        baseline = base["prediction"]
        finetuned = ft["prediction"]

        baseline_errors = edit_distance(reference, baseline)
        finetuned_errors = edit_distance(reference, finetuned)

        improvement = baseline_errors - finetuned_errors

        if improvement > 0:
            improvements.append({
                "image": base["image"],
                "reference": reference,
                "baseline": baseline,
                "finetuned": finetuned,
                "baseline_errors": baseline_errors,
                "finetuned_errors": finetuned_errors,
                "improvement": improvement,
            })

    improvements.sort(
        key=lambda x: x["improvement"],
        reverse=True,
    )

    print("\n" + "=" * 70)
    print(f"TOP {n} IMPROVEMENTS")
    print("=" * 70)

    for rank, item in enumerate(improvements[:n], 1):

        print("\n" + "-" * 70)
        print(
            f"#{rank}  {item['image']}  "
            f"({item['baseline_errors']} → "
            f"{item['finetuned_errors']} errors, "
            f"-{item['improvement']})"
        )

        print("\nREFERENCE:")
        print(item["reference"])

        print("\nBASELINE:")
        print(item["baseline"])

        print("\nFINE-TUNED:")
        print(item["finetuned"])
def main():

    baseline = load_model(BASE_MODEL, "pretrained baseline")
    baseline_results = evaluate(baseline)

    del baseline
    torch.cuda.empty_cache()

    finetuned = load_model(FINETUNED_MODEL, "fine-tuned")
    finetuned_results = evaluate(finetuned)

    del finetuned
    torch.cuda.empty_cache()
    
    print("\n" + "=" * 70)
    print("SAMPLE-BY-SAMPLE COMPARISON")
    print("=" * 70)

    baseline_correct = 0
    finetuned_correct = 0
    baseline_fixed = 0
    baseline_broken = 0

    for base, ft in zip(baseline_results, finetuned_results):

        reference = base["reference"]

        base_correct = base["prediction"] == reference
        ft_correct = ft["prediction"] == reference

        baseline_correct += base_correct
        finetuned_correct += ft_correct

        if not base_correct and ft_correct:
            baseline_fixed += 1

        if base_correct and not ft_correct:
            baseline_broken += 1

    print(f"Baseline exact matches:  {baseline_correct}/{len(baseline_results)}")
    print(f"Fine-tuned exact matches: {finetuned_correct}/{len(finetuned_results)}")
    print()
    print(f"Baseline wrong → FT correct: {baseline_fixed}")
    print(f"Baseline correct → FT wrong: {baseline_broken}")
    
    compare_predictions(
        baseline_results,
        finetuned_results,
    )
    print_best_improvements(
        baseline_results,
        finetuned_results,
        n=20,
    )


if __name__ == "__main__":
    main()

