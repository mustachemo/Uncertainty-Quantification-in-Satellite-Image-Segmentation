import torch
from pathlib import Path
import numpy as np

from model.unet_model import UNet
from utils.checker import check_dirs, check_prepped_data
from utils.visualize import visualize_prediction
from utils.logger_prep import get_logger
from configs import *

logger = get_logger(__name__)

def predict_single_model(test_loader, device):
    """
    Loads a trained model, makes predictions on a few samples from the test set,
    and visualizes the results.
    """
    # Initialize model
    model = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES, bilinear=False).to(device)

    # Load the trained model
    model_path = Path(f"checkpoints/unet_model_{DROPOUT_RATE}_{ACTIVATION_FUNC}.pth")
    if not model_path.exists():
        logger.error(f"Model checkpoint not found at {model_path}. Please train the model first.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.info(f"Model loaded from {model_path}")

    logger.info("Visualizing predictions for a few test samples...")
    with torch.no_grad():
        # Get a few samples to visualize
        for i, (images, masks) in enumerate(test_loader):
            if i >= 5: # Visualize 5 samples
                break

            images, masks = images.to(device), masks.to(device)

            # Get model prediction
            outputs = model(images)

            # The output is in logits. Use sigmoid to get probabilities and threshold to get binary mask.
            preds = torch.sigmoid(outputs) > 0.5

            # Visualize the first image in the batch
            visualize_prediction(
                images[0],
                masks[0],
                preds[0]
            )

    logger.info("Prediction visualization complete.")

if __name__ == "__main__":
    check_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load test data
    # Set shuffle=False to get consistent samples for visualization
    dataloaders = check_prepped_data(get_train=False, get_test=True)

    predict_single_model(dataloaders["test"], device)