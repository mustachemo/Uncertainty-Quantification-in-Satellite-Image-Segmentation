import torch
import numpy as np
from pathlib import Path

from configs import *
from utils.checker import check_dirs, check_prepped_data
from utils.logger_prep import get_logger
from model.unet_model import UNet

from utils.MC_dropout import (
    mc_dropout_predictions,
    visualize_mean_std,
    visualize_confidence_intervals,
    plot_correlation_analysis,
    get_uncertainty_avgs,
    run_mc_dropout_on_all_images,
)

logger = get_logger(__name__)

def run_single_image_uq(model, dataloader, device):
    """
    Performs and visualizes uncertainty quantification on a few single images.
    """
    logger.info("Running UQ analysis on single images...")
    # Get a few samples for analysis
    for i, (image, mask) in enumerate(dataloader):
        if i >= 5: # Analyze 5 samples
            break

        # Perform MC dropout inference
        mc_preds = mc_dropout_predictions(
            model, image[0], num_samples=NUM_SAMPLES_MC_DROPOUT_PREDICTION, device=device
        )

        # Calculate mean and standard deviation
        mean_prediction = torch.mean(mc_preds, dim=0)
        std_deviation = torch.std(mc_preds, dim=0)

        logger.info(f"--- Visualizing results for sample {i+1} ---")
        # Visualize results
        visualize_mean_std(
            image[0], mask[0], mean_prediction, std_deviation
        )
        visualize_confidence_intervals(
            mean_prediction, std_deviation, confidence_level=0.95
        )
        plot_correlation_analysis(mean_prediction, std_deviation)
        get_uncertainty_avgs(mean_prediction, std_deviation)


def run_full_dataset_uq(model, dataloader, device):
    """
    Performs UQ analysis on the entire dataset and analyzes the overall results.
    """
    logger.info("Running UQ analysis on the full test dataset...")
    all_means, all_stds = run_mc_dropout_on_all_images(
        model, dataloader, num_samples=NUM_SAMPLES_MC_DROPOUT_PREDICTION, device=device
    )

    # Convert lists of tensors to single tensors for analysis
    overall_mean = torch.mean(torch.stack(all_means), dim=0)
    overall_std = torch.mean(torch.stack(all_stds), dim=0)

    logger.info("--- Analyzing overall dataset uncertainty ---")
    plot_correlation_analysis(overall_mean, overall_std)
    get_uncertainty_avgs(overall_mean, overall_std)


if __name__ == "__main__":
    check_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load Data
    dataloaders = check_prepped_data(get_train=False, get_test=True)
    test_loader = dataloaders["test"]

    # Load Model
    model = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES, bilinear=False)
    model_path = Path(f"checkpoints/unet_model_{DROPOUT_RATE}_{ACTIVATION_FUNC}.pth")
    if not model_path.exists():
        logger.error(f"Model checkpoint not found at {model_path}. Please train the model first.")
        exit()
    model.load_state_dict(torch.load(model_path, map_location=device))
    logger.info(f"Model loaded from {model_path}")

    # --- Run Uncertainty Quantification Experiments ---

    # Experiment 1: Analyze a few individual images
    run_single_image_uq(model, test_loader, device)

    # Experiment 2: Analyze the entire test set (can be time-consuming)
    # run_full_dataset_uq(model, test_loader, device)

    logger.info("Uncertainty quantification experiments complete.")