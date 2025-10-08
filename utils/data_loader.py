import torch
import cv2
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging
import numpy as np
from pathlib import Path

# It's better to pass config values as arguments, but for now I'll import them to keep changes minimal
from configs import X_DIMENSION, Y_DIMENSION, BATCH_SIZE, MIXED_PRECISION

def process_image(image_path, mask_path, make_mask_binary=True):
    try:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)  # Grayscale

        if image is None or mask is None:
            logging.warning(f"Could not read image or mask for {image_path}")
            return None, None

        image = cv2.resize(image, (X_DIMENSION, Y_DIMENSION), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (X_DIMENSION, Y_DIMENSION), interpolation=cv2.INTER_LINEAR)

        # from HWC to CHW for PyTorch
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0

        if make_mask_binary:
            mask = (mask > 0.5).type(torch.uint8)

        return image, mask
    except Exception as e:
        logging.error(f"Error processing {image_path}: {e}")
        return None, None

def load_and_process_files(image_dir, mask_dir, prefix="train"):
    images = []
    masks = []
    image_paths = list(Path(image_dir).glob("*.png"))

    with ThreadPoolExecutor() as executor:
        # Create mask paths corresponding to image paths
        mask_paths = [Path(mask_dir) / f"{p.stem}_mask.png" for p in image_paths]
        futures = {executor.submit(process_image, ip, mp): ip for ip, mp in zip(image_paths, mask_paths)}

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Loading and processing {prefix} data"):
            image, mask = future.result()
            if image is not None and mask is not None:
                images.append(image)
                masks.append(mask)

    return images, masks

def save_preprocessed_data(directory, images, masks, prefix="train"):
    """Saves images and masks as individual .pt files in the specified directory."""
    os.makedirs(directory, exist_ok=True)
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(torch.save, (image, mask), os.path.join(directory, f"datapoint_{i}.pt"))
                   for i, (image, mask) in enumerate(zip(images, masks))]

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Saving preprocessed {prefix} data"):
            future.result()

class PreprocessedDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.file_list = sorted(list(self.data_dir.glob("*.pt")))

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        filepath = self.file_list[idx]
        image, mask = torch.load(filepath)
        if MIXED_PRECISION:
            image = image.half()
        # The combined_loss function expects a float tensor for y_true for BCEWithLogitsLoss
        return image, mask.float()

def create_pytorch_dataloader(data_dir, batch_size=BATCH_SIZE, shuffle=True):
    dataset = PreprocessedDataset(data_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=os.cpu_count()//2 if os.cpu_count() else 0, pin_memory=True)
    return dataloader