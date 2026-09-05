import torch
import unicodedata
from pathlib import Path
from PIL import Image

from kraken_monkey_patch import apply_kraken_monkey_patch

apply_kraken_monkey_patch()

from kraken.models import load_models
from kraken.configs import RecognitionInferenceConfig
from kraken.lib.dataset import ImageInputTransforms


# ============================================================
# MODELS
# ============================================================

MODELS = {
    "Base": (
        "/home/gnikoul/.local/share/htrmopo/"
        "a328a8d1-29fb-519a-bf81-3f56a512f080/"
        "ppocr_v6_tau090.safetensors"
    ),

    "EPARCHOS": (
        "models/eparchos_ppocrv6_nfd/"
        "best_0.9637.safetensors"
    ),

    "XF53": (
        "models/xf53_ppocrv6/"
        "best_0.9040.safetensors"
    ),
    "EPARCHOS+XF53": (
        "models/combined_ppocrv6/"
        "best_0.8614.safetensors"
    ),
}


# ============================================================
# TEST DATASETS
# ============================================================

DATASETS = {
    "EPARCHOS": {
        "image_dir": Path("datasets/eparchos/training_nfd/test"),
        "label_dir": Path("datasets/eparchos/training_nfd/test"),
    },

    "XF53": {
        "image_dir": Path("datasets/XF53/test"),
        "label_dir": Path("datasets/XF53/test_normalized"),
    },
}


# ============================================================
# HELPERS
# ============================================================

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


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, dataset_name):

    dataset = DATASETS[dataset_name]

    image_dir = dataset["image_dir"]
    label_dir = dataset["label_dir"]

    print("\n" + "=" * 70)
    print(f"EVALUATING ON {dataset_name.upper()} TEST SET")
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

    results = []

    image_paths = sorted(image_dir.glob("*.png"))

    print(f"Test samples: {len(image_paths)}")
    print("Running recognition...\n")

    for i, image_path in enumerate(image_paths, 1):

        label_path = label_dir / f"{image_path.stem}.gt.txt"

        if not label_path.exists():
            raise FileNotFoundError(
                f"Missing label: {label_path}"
            )

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

        results.append({
            "image": image_path.name,
            "reference": reference,
            "prediction": prediction,
        })

        # ----------------------------------------------------
        # CER
        # ----------------------------------------------------

        char_errors = edit_distance(
            reference,
            prediction,
        )

        char_count = len(reference)

        total_char_errors += char_errors
        total_chars += char_count

        # ----------------------------------------------------
        # WER
        # ----------------------------------------------------

        ref_words = reference.split()
        pred_words = prediction.split()

        word_errors = edit_distance(
            ref_words,
            pred_words,
        )

        word_count = len(ref_words)

        total_word_errors += word_errors
        total_words += word_count

        if i % 25 == 0:
            print(f"Processed {i}/{len(image_paths)}")

    cer = total_char_errors / total_chars
    wer = total_word_errors / total_words

    print("\n" + "=" * 60)
    print(f"{dataset_name.upper()} RESULTS")
    print("=" * 60)

    print(f"Characters:       {total_chars}")
    print(f"Character errors: {total_char_errors}")
    print(f"CER:              {cer:.4f} ({cer * 100:.2f}%)")
    print()

    print(f"Words:            {total_words}")
    print(f"Word errors:      {total_word_errors}")
    print(f"WER:              {wer:.4f} ({wer * 100:.2f}%)")

    return {
        "results": results,
        "cer": cer,
        "wer": wer,
        "char_errors": total_char_errors,
        "chars": total_chars,
        "word_errors": total_word_errors,
        "words": total_words,
    }


# ============================================================
# MODEL COMPARISON
# ============================================================

def compare_predictions(
    baseline_results,
    comparison_results,
    baseline_name,
    comparison_name,
):

    fixed_errors = []
    broken_errors = []

    for base, comp in zip(
        baseline_results,
        comparison_results,
    ):

        reference = base["reference"]

        base_prediction = base["prediction"]
        comp_prediction = comp["prediction"]

        base_distance = edit_distance(
            reference,
            base_prediction,
        )

        comp_distance = edit_distance(
            reference,
            comp_prediction,
        )

        if comp_distance < base_distance:

            fixed_errors.append({
                "image": base["image"],
                "reference": reference,
                "baseline": base_prediction,
                "comparison": comp_prediction,
                "baseline_errors": base_distance,
                "comparison_errors": comp_distance,
            })

        elif comp_distance > base_distance:

            broken_errors.append({
                "image": base["image"],
                "reference": reference,
                "baseline": base_prediction,
                "comparison": comp_prediction,
                "baseline_errors": base_distance,
                "comparison_errors": comp_distance,
            })

    print("\n" + "=" * 70)
    print(f"{baseline_name} → {comparison_name}")
    print("=" * 70)

    print(f"Images improved:  {len(fixed_errors)}")
    print(f"Images worsened:  {len(broken_errors)}")

    total_fixed = sum(
        x["baseline_errors"] - x["comparison_errors"]
        for x in fixed_errors
    )

    total_broken = sum(
        x["comparison_errors"] - x["baseline_errors"]
        for x in broken_errors
    )

    print(f"Errors eliminated: {total_fixed}")
    print(f"Errors introduced:  {total_broken}")


def print_best_improvements(
    baseline_results,
    comparison_results,
    baseline_name,
    comparison_name,
    n=10,
):

    improvements = []

    for base, comp in zip(
        baseline_results,
        comparison_results,
    ):

        reference = base["reference"]

        baseline = base["prediction"]
        comparison = comp["prediction"]

        baseline_errors = edit_distance(
            reference,
            baseline,
        )

        comparison_errors = edit_distance(
            reference,
            comparison,
        )

        improvement = (
            baseline_errors - comparison_errors
        )

        if improvement > 0:

            improvements.append({
                "image": base["image"],
                "reference": reference,
                "baseline": baseline,
                "comparison": comparison,
                "baseline_errors": baseline_errors,
                "comparison_errors": comparison_errors,
                "improvement": improvement,
            })

    improvements.sort(
        key=lambda x: x["improvement"],
        reverse=True,
    )

    print("\n" + "=" * 70)
    print(
        f"TOP {n} IMPROVEMENTS: "
        f"{baseline_name} → {comparison_name}"
    )
    print("=" * 70)

    for rank, item in enumerate(
        improvements[:n],
        1,
    ):

        print("\n" + "-" * 70)

        print(
            f"#{rank}  {item['image']}  "
            f"({item['baseline_errors']} → "
            f"{item['comparison_errors']} errors, "
            f"-{item['improvement']})"
        )

        print("\nREFERENCE:")
        print(item["reference"])

        print(f"\n{baseline_name.upper()}:")
        print(item["baseline"])

        print(f"\n{comparison_name.upper()}:")
        print(item["comparison"])


# ============================================================
# SUMMARY TABLE
# ============================================================

def print_summary(all_results):

    print("\n\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(
        f"{'Model':<15}"
        f"{'Dataset':<15}"
        f"{'CER':>12}"
        f"{'WER':>12}"
    )

    print("-" * 54)

    for model_name in MODELS:

        for dataset_name in DATASETS:

            result = all_results[
                model_name
            ][dataset_name]

            print(
                f"{model_name:<15}"
                f"{dataset_name:<15}"
                f"{result['cer'] * 100:>11.2f}%"
                f"{result['wer'] * 100:>11.2f}%"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    all_results = {}

    for model_name, model_path in MODELS.items():

        all_results[model_name] = {}

        model = load_model(
            model_path,
            model_name,
        )

        for dataset_name in DATASETS:

            all_results[model_name][dataset_name] = (
                evaluate(
                    model,
                    dataset_name,
                )
            )

        del model
        torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Comparisons
    # --------------------------------------------------------

    for dataset_name in DATASETS:

        print("\n\n" + "=" * 80)
        print(f"COMPARISONS ON {dataset_name.upper()}")
        print("=" * 80)

        base_results = all_results["Base"][dataset_name]["results"]

        for comparison_name in [
            "EPARCHOS",
            "XF53",
        ]:

            comparison_results = all_results[
                comparison_name
            ][dataset_name]["results"]

            compare_predictions(
                base_results,
                comparison_results,
                "Base",
                comparison_name,
            )

        # Direct EPARCHOS vs XF53 comparison
        compare_predictions(
            all_results["EPARCHOS"][dataset_name]["results"],
            all_results["XF53"][dataset_name]["results"],
            "EPARCHOS",
            "XF53",
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_summary(all_results)


if __name__ == "__main__":
    main()