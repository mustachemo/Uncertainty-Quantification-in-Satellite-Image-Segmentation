import torch
from tqdm import tqdm
from pathlib import Path

from model.unet_model import UNet
from utils.checker import check_dirs, check_prepped_data
from utils.logger_prep import get_logger
from configs import *
from utils.custom_funcs import dice_coefficient

logger = get_logger(__name__)

def evaluate_model(model, dataloader, device):
    """
    Evaluates the model on a given dataloader.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (DataLoader): The dataloader for evaluation data.
        device (torch.device): The device to run evaluation on.

    Returns:
        float: The average Dice score over the dataset.
    """
    model.eval()
    total_dice = 0
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            dice = dice_coefficient(outputs, masks).item()
            total_dice += dice

    avg_dice = total_dice / len(dataloader)
    return avg_dice

if __name__ == "__main__":
    check_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data
    dataloaders = check_prepped_data(get_train=False, get_test=True)
    test_loader = dataloaders["test"]

    # Initialize model
    model = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES, bilinear=False).to(device)

    # Load trained model
    model_path = Path(f"checkpoints/unet_model_{DROPOUT_RATE}_{ACTIVATION_FUNC}.pth")
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"Loaded model from {model_path}")
    else:
        logger.error(f"Model checkpoint not found at {model_path}. Please train the model first.")
        exit()

    # Evaluate
    dice_score = evaluate_model(model, test_loader, device)
    logger.info(f"Final Dice Score on Test Set: {dice_score}")