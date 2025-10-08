import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import mlflow
import mlflow.pytorch
import numpy as np
from pathlib import Path
import os

from model.unet_model import UNet
from utils.visualize import visualize_sample_with_mask
from utils.logger_prep import get_logger
from utils.custom_funcs import combined_loss, dice_coefficient
from utils.checker import check_dirs, check_prepped_data
from configs import *

logger = get_logger(__name__)

def train_unet(train_loader, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES, bilinear=False).to(device)
    
    # MLflow setup
    mlflow.set_experiment("Satellite-UNet-PyTorch")
    with mlflow.start_run() as run:
        mlflow.log_params({
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "dropout_rate": DROPOUT_RATE,
            "mixed_precision": MIXED_PRECISION,
            "activation_function": ACTIVATION_FUNC,
            "n_channels": N_CHANNELS,
            "n_classes": N_CLASSES
        })

        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        scaler = torch.cuda.amp.GradScaler(enabled=MIXED_PRECISION)

        best_val_dice = 0.0
        model_path = Path(f"checkpoints/unet_model_{DROPOUT_RATE}_{ACTIVATION_FUNC}.pth")

        for epoch in range(EPOCHS):
            model.train()
            epoch_loss = 0

            with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch") as pbar:
                for images, masks in train_loader:
                    images, masks = images.to(device), masks.to(device)

                    with torch.cuda.amp.autocast(enabled=MIXED_PRECISION):
                        outputs = model(images)
                        loss = combined_loss(outputs, masks)

                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    epoch_loss += loss.item()
                    pbar.update(1)
                    pbar.set_postfix(**{'loss (batch)': loss.item()})

            # Validation
            val_dice = evaluate(model, val_loader, device)
            logger.info(f"Epoch {epoch+1}, Validation Dice Score: {val_dice}")
            mlflow.log_metric("validation_dice_score", val_dice, step=epoch)
            mlflow.log_metric("training_loss", epoch_loss / len(train_loader), step=epoch)

            if val_dice > best_val_dice:
                best_val_dice = val_dice
                torch.save(model.state_dict(), model_path)
                logger.info(f"Model saved to {model_path}")
                mlflow.pytorch.log_model(model, "model")


def evaluate(model, dataloader, device):
    model.eval()
    total_dice = 0
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            total_dice += dice_coefficient(outputs, masks).item()
    return total_dice / len(dataloader)


if __name__ == "__main__":
    check_dirs()
    dataloaders = check_prepped_data(get_train=True, get_test=True)
    
    if not torch.cuda.is_available():
        logger.warning("No GPU found, model may be slow or fail to train")
    else:
        logger.info("GPU found!")

    train_unet(dataloaders["train"], dataloaders["test"])