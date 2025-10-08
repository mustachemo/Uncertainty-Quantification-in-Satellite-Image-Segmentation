import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch

from .logger_prep import get_logger
from .custom_funcs import dice_coefficient
from .visualize import tensor_to_image

logger = get_logger(__name__)

def enable_dropout(model):
    """Function to enable the dropout layers during test-time"""
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def mc_dropout_predictions(model, input_image, num_samples=50, device='cpu'):
    """
    Perform MC dropout inference to get a distribution of predictions.

    Args:
        model (torch.nn.Module): Trained model with dropout layers.
        input_image (torch.Tensor): Input image tensor for predictions.
        num_samples (int): Number of stochastic forward passes.
        device (torch.device): The device to run inference on.

    Returns:
        torch.Tensor: Tensor of predictions from each forward pass.
    """
    model.to(device)
    enable_dropout(model) # Make sure dropout is active

    predictions = []
    input_image = input_image.unsqueeze(0).to(device) # Add batch dimension and send to device

    with torch.no_grad():
        for _ in tqdm(range(num_samples), desc="MC Dropout Inference for a single image", leave=False):
            output = model(input_image)
            # Apply sigmoid to get probabilities
            output_prob = torch.sigmoid(output)
            predictions.append(output_prob.cpu())

    return torch.cat(predictions, dim=0)


def run_mc_dropout_on_all_images(model, dataloader, num_samples=10, device='cpu'):
    """
    Run MC dropout prediction on all test images multiple times.

    Args:
        model: The loaded PyTorch model with dropout.
        dataloader: DataLoader for the test set.
        num_samples: Number of Monte Carlo samples to generate per image.
        device: The device to run inference on.

    Returns:
        Tuple of lists: (all_mean_predictions, all_std_deviations)
    """
    all_mean_predictions = []
    all_std_deviations = []

    model.to(device)
    enable_dropout(model)

    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc=f"MC Dropout on all images ({num_samples} samples each)"):
            images = images.to(device)

            batch_predictions = []
            for _ in range(num_samples):
                outputs = model(images)
                batch_predictions.append(torch.sigmoid(outputs).cpu())

            mc_samples = torch.stack(batch_predictions) # Shape: (num_samples, batch_size, C, H, W)

            mean_prediction = torch.mean(mc_samples, dim=0)
            std_deviation = torch.std(mc_samples, dim=0)

            all_mean_predictions.extend(list(torch.unbind(mean_prediction, dim=0)))
            all_std_deviations.extend(list(torch.unbind(std_deviation, dim=0)))

    return all_mean_predictions, all_std_deviations


def visualize_mean_std(test_image, test_mask, mean_prediction, std_deviation):
    """Visualize the test image, mask, mean prediction, and uncertainty."""

    # Ensure inputs are suitable for visualization
    img_np = tensor_to_image(test_image)
    mask_np = tensor_to_image(test_mask)
    mean_np = tensor_to_image(mean_prediction)
    std_np = tensor_to_image(std_deviation)

    # Calculate Dice score (assuming mean_prediction is probability)
    dice = dice_coefficient(torch.tensor(mean_prediction).unsqueeze(0), torch.tensor(test_mask).unsqueeze(0))

    plt.figure(figsize=(12, 12))

    plt.subplot(2, 2, 1)
    plt.imshow(img_np)
    plt.title("Test Image")
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(mask_np, cmap="gray")
    plt.title("Ground Truth Mask")
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(mean_np, cmap="gray")
    plt.colorbar()
    plt.title("Mean Prediction")
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(std_np, cmap="gray")
    plt.colorbar()
    plt.title("Prediction Uncertainty (Std Dev)")
    plt.axis('off')

    plt.suptitle(f"Dice Coefficient: {dice.item():.4f}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def visualize_confidence_intervals(mean_prediction, std_deviation, confidence_level=0.95):
    """Visualize the confidence intervals of the model predictions."""
    if isinstance(mean_prediction, torch.Tensor):
        mean_prediction = mean_prediction.cpu().numpy()
    if isinstance(std_deviation, torch.Tensor):
        std_deviation = std_deviation.cpu().numpy()

    z_score = 1.96  # for 95% confidence
    lower_bound = mean_prediction - z_score * std_deviation
    upper_bound = mean_prediction + z_score * std_deviation

    lower_bound = np.squeeze(lower_bound)
    upper_bound = np.squeeze(upper_bound)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(lower_bound, cmap="gray", vmin=0, vmax=1)
    plt.colorbar()
    plt.title(f"{confidence_level * 100}% Confidence Lower Bound")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(upper_bound, cmap="gray", vmin=0, vmax=1)
    plt.colorbar()
    plt.title(f"{confidence_level * 100}% Confidence Upper Bound")
    plt.axis('off')

    plt.show()


def plot_correlation_analysis(mean_prediction, std_deviation):
    """Plot correlation between mean predictions and their standard deviations."""
    if isinstance(mean_prediction, torch.Tensor):
        mean_prediction = mean_prediction.cpu().numpy()
    if isinstance(std_deviation, torch.Tensor):
        std_deviation = std_deviation.cpu().numpy()

    flat_mean = mean_prediction.flatten()
    flat_std = std_deviation.flatten()

    plt.figure(figsize=(6, 6))
    plt.scatter(flat_mean, flat_std, alpha=0.1)
    plt.xlabel("Mean Prediction")
    plt.ylabel("Standard Deviation")
    plt.title("Correlation between Mean Prediction and Uncertainty")
    plt.grid(True)
    plt.show()


def get_uncertainty_avgs(mean_prediction, std_deviation):
    """Calculate and log average uncertainty for different prediction confidence levels."""
    if isinstance(mean_prediction, torch.Tensor):
        mean_prediction = mean_prediction.cpu().numpy()
    if isinstance(std_deviation, torch.Tensor):
        std_deviation = std_deviation.cpu().numpy()

    flat_mean = mean_prediction.flatten()
    flat_std = std_deviation.flatten()

    # Define bins for low, medium, and high confidence predictions
    low_conf_uncertainty = flat_std[flat_mean < 0.2]
    medium_conf_uncertainty = flat_std[(flat_mean >= 0.2) & (flat_mean < 0.8)]
    high_conf_uncertainty = flat_std[flat_mean >= 0.8]

    logger.info(f"Avg uncertainty for low-confidence predictions (<0.2): {np.mean(low_conf_uncertainty):.4f}")
    logger.info(f"Avg uncertainty for mid-confidence predictions (0.2-0.8): {np.mean(medium_conf_uncertainty):.4f}")
    logger.info(f"Avg uncertainty for high-confidence predictions (>0.8): {np.mean(high_conf_uncertainty):.4f}")