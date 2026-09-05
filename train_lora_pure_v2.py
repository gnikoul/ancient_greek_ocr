import os

import glob

from peft import LoraConfig, get_peft_model
from kraken.models import load_models
from kraken.train.ppocr import (
    PPOCRv6RecognitionModel,
    PPOCRv6RecognitionDataModule,
)
from kraken.configs.ppocr import (
    PPOCRv6RecognitionTrainingConfig,
    PPOCRv6RecognitionTrainingDataConfig,
)
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint


BASE_MODEL = (
    "/home/gnikoul/.local/share/htrmopo/"
    "a328a8d1-29fb-519a-bf81-3f56a512f080/"
    "ppocr_v6_tau090.safetensors"
)

TRAIN_DATA = "datasets/eparchos/training/train"
VAL_DATA = "datasets/eparchos/training/val"

print("Creating data module...")

train_images = glob.glob(os.path.join(TRAIN_DATA, "*.png"))
val_images = glob.glob(os.path.join(VAL_DATA, "*.png"))

print(f"Training images:   {len(train_images)}")
print(f"Validation images: {len(val_images)}")

data_config = PPOCRv6RecognitionTrainingDataConfig(
    training_data=train_images,
    evaluation_data=val_images,
    format_type="path",
    padding=16,
    max_width=2560,
)

OUTPUT_DIR = "models/eparchos_ppocrv6_lora_cosine"


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

import torch


class LoRAPPOCRv6RecognitionModel(PPOCRv6RecognitionModel):

    def configure_optimizers(self):
        lora_params = [
            param
            for name, param in self.named_parameters()
            if "lora_" in name
        ]

        total = sum(p.numel() for p in lora_params)

        print(f"\nOptimizer will train ONLY {total:,} LoRA parameters")

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
    
def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------
    # Load pretrained PP-OCRv6
    # ------------------------------------------------------------

    print("Loading PP-OCRv6...")

    model = load_models(BASE_MODEL)[0]

    # ------------------------------------------------------------
    # Add LoRA
    # ------------------------------------------------------------

    print("Adding LoRA...")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=LORA_TARGETS,
        bias="none",
    )

    model.nn = get_peft_model(model.nn, lora_config)

    # Freeze everything except LoRA parameters.
    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    print()
    print(f"Trainable parameters: {trainable:,}")
    print(f"Total parameters:     {total:,}")
    print(f"Trainable percentage: {100 * trainable / total:.4f}%")

    # ------------------------------------------------------------
    # Training configuration
    # ------------------------------------------------------------

    config = PPOCRv6RecognitionTrainingConfig(
        resize="union",
        batch_size=8,
        lrate=1e-4,
        optimizer="AdamW",
        schedule="cosine",
        epochs=10,
        augment=True,
        output=OUTPUT_DIR,
    )

    # ------------------------------------------------------------
    # Data configuration
    # ------------------------------------------------------------


    # ------------------------------------------------------------
    # Kraken training model
    # ------------------------------------------------------------

    print("\nCreating Kraken training model...")

    training_model = LoRAPPOCRv6RecognitionModel(
        config=config,
        model=model,
    )
    
    # Make absolutely sure only LoRA parameters are trainable.
    for name, param in training_model.named_parameters():
        param.requires_grad = "lora_" in name


    print("\nTrainable parameters:")

    trainable = 0

    for name, param in training_model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {param.numel():,}")
            trainable += param.numel()

    print(f"\nTotal trainable: {trainable:,}")

    # ------------------------------------------------------------
    # Data module
    # ------------------------------------------------------------

    print("\nCreating data module...")

    datamodule = PPOCRv6RecognitionDataModule(
        data_config=data_config,
    )

    # ------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------

    checkpoint = ModelCheckpoint(
        dirpath=OUTPUT_DIR,
        filename="lora-{epoch:02d}-{val_accuracy:.4f}",
        monitor="val_accuracy",
        mode="max",
        save_top_k=1,
    )

    # ------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=config.epochs,
        callbacks=[checkpoint],
        precision="16-mixed",
        log_every_n_steps=10,
    )

    # ------------------------------------------------------------
    # Train
    # ------------------------------------------------------------

    print("\nStarting LoRA training...")
    print(f"Output directory: {OUTPUT_DIR}")

    trainer.fit(
        training_model,
        datamodule=datamodule,
    )

    print("\nTraining complete.")
    print("Best checkpoint:")
    print(checkpoint.best_model_path)


if __name__ == "__main__":
    main()