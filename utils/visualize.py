import json
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from pathlib import Path
import pandas as pd
import glob
import torch
import numpy as np

def load_image_and_mask(image_index):
    """
    Load image and mask based on the image index.
    """
    image_path = Path(r"data/images/val") / f"img_resize_{image_index}.png"
    mask_path = Path(r"data/mask/val") / f"img_resize_{image_index}_mask.png"

    try:
        image = Image.open(image_path)
        mask = Image.open(mask_path)
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        exit()

    return image, mask


def load_bboxes():
    """
    Load bounding boxes from all_bbox.txt.
    """
    try:
        with open(r"data\all_bbox.txt") as f:
            bboxes = json.load(f)
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        exit()

    return bboxes


def draw_bboxes_on_image(image, bboxes):
    """
    Draw bounding boxes on the image.
    """
    draw = ImageDraw.Draw(image)
    for bbox in bboxes:
        draw.rectangle([bbox[2], bbox[3], bbox[0], bbox[1]], outline="yellow", width=3)

    return image


def tensor_to_image(tensor):
    """Convert a PyTorch tensor to a NumPy image for visualization."""
    if tensor.is_cuda:
        tensor = tensor.cpu()
    # If the tensor has a channel dimension, move it to the last axis
    if tensor.dim() == 3 and tensor.shape[0] in [1, 3, 4]:
        tensor = tensor.permute(1, 2, 0)
    # Squeeze singleton dimensions
    tensor = tensor.squeeze()
    # Convert to numpy
    img_np = tensor.numpy()
    # If it's a normalized image, scale it back to 0-255
    if img_np.max() <= 1.0:
        img_np = (img_np * 255).astype(np.uint8)
    return img_np


def visualize_sample_with_mask(image, mask):
    """Visualize a sample image and its mask, handles both tensors and numpy arrays."""
    if isinstance(image, torch.Tensor):
        image = tensor_to_image(image)
    if isinstance(mask, torch.Tensor):
        mask = tensor_to_image(mask)

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Mask")
    plt.axis("off")
    plt.show()
    plt.close()


def visualize_prediction(test_image, test_mask, prediction_mask):
    """
    Visualize a sample image, its ground truth mask, and the model's prediction.
    Handles both tensors and numpy arrays.
    """
    if isinstance(test_image, torch.Tensor):
        test_image = tensor_to_image(test_image)
    if isinstance(test_mask, torch.Tensor):
        test_mask = tensor_to_image(test_mask)
    if isinstance(prediction_mask, torch.Tensor):
        prediction_mask = tensor_to_image(prediction_mask)

    fig, ax = plt.subplots(1, 3, figsize=(15, 7))

    ax[0].imshow(test_image)
    ax[0].set_title("Image")
    ax[1].imshow(test_mask, cmap="gray")
    ax[1].set_title("Ground Truth Mask")
    ax[2].imshow(prediction_mask, cmap="gray")
    ax[2].set_title("Predicted Mask")

    plt.tight_layout()
    plt.show()


def visualize_training_logs():
    log_files = glob.glob("logs/model_*.log")
    data = {}

    for log_file in log_files:
        try:
            # Attempt to extract a meaningful name, e.g., from dropout/activation
            parts = log_file.split('_')
            name = f"D{parts[2]}_{parts[3]}"
            df = pd.read_csv(log_file)
            if "val_dice_coefficient" in df.columns:
                data[name] = df["val_dice_coefficient"]
        except Exception:
            # Fallback for different naming conventions
            df = pd.read_csv(log_file)
            if "validation_dice_score" in df.columns:
                data[log_file] = df["validation_dice_score"]


    plt.figure(figsize=(10, 6))

    for name, val_dice_coefs in data.items():
        plt.plot(val_dice_coefs, label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Validation Dice Score")
    plt.title("Validation Dice Score vs. Epoch")
    plt.legend()
    plt.grid(True)
    plt.show()