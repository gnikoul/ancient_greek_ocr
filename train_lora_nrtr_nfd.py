import torch

from pathlib import Path

from peft import LoraConfig, get_peft_model

from kraken_monkey_patch import apply_kraken_monkey_patch

apply_kraken_monkey_patch()

from kraken.models import load_models
from kraken.train.ppocr import (
    PPOCRv6RecognitionDataModule,
    PPOCRv6RecognitionModel,
)

from kraken.configs.ppocr import (
    PPOCRv6RecognitionTrainingConfig,
    PPOCRv6RecognitionTrainingDataConfig,
)


MODEL = (
    "/home/gnikoul/.local/share/htrmopo/"
    "a328a8d1-29fb-519a-bf81-3f56a512f080/"
    "ppocr_v6_tau090.safetensors"
)

TRAIN_DATA = "datasets/eparchos/training_nfd/train"
VAL_DATA = "datasets/eparchos/training_nfd/val"

OUTPUT_DIR = "models/eparchos_ppocrv6_lora_nrtr_nfd"

EPOCHS = 10
BATCH_SIZE = 8
LR = 1e-4


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


class LoRANRTRRecognitionModel(PPOCRv6RecognitionModel):

    def configure_optimizers(self):
        # LoRA parameters from the recognition network.
        lora_params = [
            param
            for name, param in self.named_parameters()
            if "lora_" in name
        ]

        # Kraken builds this during setup().
        if self.nrtr_head is None:
            raise RuntimeError(
                "NRTR head has not been built before configure_optimizers()."
            )

        nrtr_params = list(self.nrtr_head.parameters())

        trainable_params = lora_params + nrtr_params

        lora_total = sum(p.numel() for p in lora_params)
        nrtr_total = sum(p.numel() for p in nrtr_params)
        total = lora_total + nrtr_total

        print()
        print("=" * 60)
        print("LoRA + NRTR optimizer")
        print("=" * 60)
        print(f"LoRA parameters:  {lora_total:,}")
        print(f"NRTR parameters: {nrtr_total:,}")
        print(f"Total trainable: {total:,}")
        print("=" * 60)
        print()

        optimizer = torch.optim.AdamW(
            trainable_params,
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

    print("=" * 70)
    print("NFD LoRA + NRTR TRAINING")
    print("=" * 70)

    print("\nLoading pretrained model...")

    model = load_models(MODEL)[0]

    print("Pretrained codec size:", len(model.codec))

    # --------------------------------------------------
    # LoRA configuration
    # --------------------------------------------------

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=LORA_TARGETS,
        bias="none",
    )

    model.nn = get_peft_model(
        model.nn,
        lora_config,
    )

    model.nn.print_trainable_parameters()

    # --------------------------------------------------
    # Training configuration
    # --------------------------------------------------

    data_config = PPOCRv6RecognitionTrainingDataConfig(
        training_data=[
            str(p)
            for p in Path(TRAIN_DATA).glob("*.png")
        ],
        evaluation_data=[
            str(p)
            for p in Path(VAL_DATA).glob("*.png")
        ],
        format_type="path",
        padding=16,
        max_width=2560,
    )

    config = PPOCRv6RecognitionTrainingConfig(
        resize="fail",
        batch_size=BATCH_SIZE,
        lrate=LR,
        optimizer="AdamW",
        schedule="cosine",
        epochs=EPOCHS,
        augment=True,
        output=OUTPUT_DIR,
    )

    # --------------------------------------------------
    # Data module
    # --------------------------------------------------

    data = PPOCRv6RecognitionDataModule(data_config)

    print("\nTraining images:", len(data.train_set))
    print("Validation images:", len(data.val_set))

    # --------------------------------------------------
    # Create Kraken training model.
    #
    # The NRTR head is created by the Kraken training
    # model and remains trainable.
    # --------------------------------------------------

    recognition_model = LoRANRTRRecognitionModel(
        config=config,
        model=model,
    )

    # --------------------------------------------------
    # Output directory
    # --------------------------------------------------

    Path(OUTPUT_DIR).mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Checkpoint callback
    # --------------------------------------------------

    from lightning.pytorch.callbacks import ModelCheckpoint

    checkpoint_callback = ModelCheckpoint(
        dirpath=OUTPUT_DIR,
        filename="lora-nrtr-nfd-{epoch:02d}-{val_accuracy:.4f}",
        monitor="val_accuracy",
        mode="max",
        save_top_k=1,
    )

    # --------------------------------------------------
    # Lightning trainer
    # --------------------------------------------------

    from lightning.pytorch import Trainer

    trainer = Trainer(
        accelerator="cuda",
        devices=1,

        max_epochs=EPOCHS,

        callbacks=[
            checkpoint_callback,
        ],

        default_root_dir=OUTPUT_DIR,

        log_every_n_steps=10,
    )

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    trainer.fit(
        recognition_model,
        datamodule=data,
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        "Best checkpoint:",
        checkpoint_callback.best_model_path,
    )

    print(
        "Best val_accuracy:",
        checkpoint_callback.best_model_score,
    )


if __name__ == "__main__":
    main()