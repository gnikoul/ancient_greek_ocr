import glob
import os

import torch
from peft import LoraConfig, get_peft_model
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from kraken.models import load_models
from kraken.train.ppocr import (
    PPOCRv6RecognitionDataModule,
    PPOCRv6RecognitionModel,
)
from kraken.configs.ppocr import (
    PPOCRv6RecognitionTrainingConfig,
    PPOCRv6RecognitionTrainingDataConfig,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_MODEL = (
    "/home/gnikoul/.local/share/htrmopo/"
    "a328a8d1-29fb-519a-bf81-3f56a512f080/"
    "ppocr_v6_tau090.safetensors"
)

TRAIN_DATA = "datasets/eparchos/training_nfd/train"
VAL_DATA = "datasets/eparchos/training_nfd/val"

OUTPUT_DIR = "models/eparchos_ppocrv6_lora_nfd"


# ---------------------------------------------------------------------------
# LoRA configuration
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# LoRA model
# ---------------------------------------------------------------------------

class LoRAPPOCRv6RecognitionModel(PPOCRv6RecognitionModel):

    def configure_optimizers(self):
        # ONLY optimize LoRA parameters.
        lora_params = [
            param
            for name, param in self.named_parameters()
            if "lora_" in name
        ]

        total = sum(param.numel() for param in lora_params)

        print()
        print("=" * 60)
        print("LoRA optimizer")
        print("=" * 60)
        print(f"Trainable LoRA parameters: {total:,}")
        print("=" * 60)
        print()

        optimizer = torch.optim.AdamW(
            lora_params,
            lr=self.hparams.config.lrate,
            weight_decay=0.01,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.config.epochs,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---------------------------------------------------------------
    # Load pretrained model
    # ---------------------------------------------------------------

    print("Loading pretrained model...")
    model = load_models(BASE_MODEL)[0]

    original_codec_size = len(model.codec.c2l)

    print(f"Pretrained codec size: {original_codec_size}")

    if original_codec_size != 1629:
        raise RuntimeError(
            f"Unexpected pretrained codec size: "
            f"{original_codec_size} (expected 1629)"
        )

    # ---------------------------------------------------------------
    # Apply LoRA
    # ---------------------------------------------------------------

    lora_config = LoraConfig(
        r=32,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=LORA_TARGETS,
    )

    model.nn = get_peft_model(
        model.nn,
        lora_config,
    )

    print()
    model.nn.print_trainable_parameters()
    print()

    # Make sure LoRA injection did not alter the codec.
    if len(model.codec.c2l) != original_codec_size:
        raise RuntimeError(
            "Codec changed after LoRA injection!"
        )

    # ---------------------------------------------------------------
    # Training data
    # ---------------------------------------------------------------

    train_images = glob.glob(
        os.path.join(TRAIN_DATA, "*.png")
    )

    val_images = glob.glob(
        os.path.join(VAL_DATA, "*.png")
    )

    print(f"Training images:   {len(train_images)}")
    print(f"Validation images: {len(val_images)}")

    if len(train_images) != 1817:
        raise RuntimeError(
            f"Expected 1817 training images, "
            f"found {len(train_images)}"
        )

    if len(val_images) != 227:
        raise RuntimeError(
            f"Expected 227 validation images, "
            f"found {len(val_images)}"
        )

    data_config = PPOCRv6RecognitionTrainingDataConfig(
        training_data=train_images,
        evaluation_data=val_images,
        format_type="path",
        padding=16,
        max_width=2560,
    )

    # ---------------------------------------------------------------
    # Training configuration
    # ---------------------------------------------------------------

    config = PPOCRv6RecognitionTrainingConfig(
        resize="fail",
        batch_size=8,
        lrate=1e-4,
        optimizer="AdamW",
        schedule="cosine",
        epochs=10,
        augment=True,
        output=OUTPUT_DIR,
    )

    # ---------------------------------------------------------------
    # Data module
    # ---------------------------------------------------------------

    datamodule = PPOCRv6RecognitionDataModule(
        data_config
    )

    # ---------------------------------------------------------------
    # Kraken Lightning model
    # ---------------------------------------------------------------

    training_model = LoRAPPOCRv6RecognitionModel(
        config=config,
        model=model
    )

    # ---------------------------------------------------------------
    # Checkpointing
    # ---------------------------------------------------------------

    checkpoint = ModelCheckpoint(
        dirpath=OUTPUT_DIR,
        filename="lora-nfd-{epoch:02d}-{val_accuracy:.4f}",
        monitor="val_accuracy",
        mode="max",
        save_top_k=1,
    )

    # ---------------------------------------------------------------
    # Trainer
    # ---------------------------------------------------------------

    trainer = Trainer(
        accelerator="cuda",
        devices=1,
        max_epochs=config.epochs,
        callbacks=[checkpoint],
        default_root_dir=OUTPUT_DIR,
    )

    # ---------------------------------------------------------------
    # Train
    # ---------------------------------------------------------------

    print()
    print("=" * 60)
    print("Starting NFD LoRA training")
    print("=" * 60)
    print()

    trainer.fit(
        training_model,
        datamodule=datamodule,
    )

    print()
    print("=" * 60)
    print("Training complete")
    print("=" * 60)
    print(f"Best checkpoint: {checkpoint.best_model_path}")
    print(f"Best val_accuracy: {checkpoint.best_model_score}")
    print()


if __name__ == "__main__":
    main()