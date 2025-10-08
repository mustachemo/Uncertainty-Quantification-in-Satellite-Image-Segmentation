from utils.data_loader import (
    load_and_process_files,
    save_preprocessed_data,
    create_pytorch_dataloader,
)
from configs import *
from pathlib import Path
import os

def check_dirs():
    """Check if directories exist, if not create them"""
    directories = [
        "checkpoints",
        "data",
        "prepped_data",
        "model",
        "logs",
        PREPPED_TRAIN_DATASET_DIR,
        PREPPED_TEST_DATASET_DIR
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def check_prepped_data(get_train=True, get_test=True):
    """Check if prepped data exists, if not create it, then return dataloaders."""
    print("--" * 20)

    # Define paths for raw and preprocessed data
    paths = {
        "train": (PREPPED_TRAIN_DATASET_DIR, TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR),
        "test": (PREPPED_TEST_DATASET_DIR, TEST_IMAGES_DIR, TEST_MASKS_DIR),
    }

    dataloaders = dict()

    # Helper function to process a dataset split (train or test)
    def process_split(split_name):
        prepped_dir, images_dir, masks_dir = paths[split_name]
        
        # Check if preprocessed data exists
        if not any(Path(prepped_dir).iterdir()):
            print(f"Preprocessed {split_name} data not found. Processing raw data...")
            images, masks = load_and_process_files(
                Path(images_dir), Path(masks_dir), prefix=split_name
            )
            save_preprocessed_data(prepped_dir, images, masks, prefix=split_name)
        else:
            print(f"Found preprocessed {split_name} data.")

        # Create a dataloader for the split
        print(f"Creating PyTorch Dataloader for {split_name} data...")
        return create_pytorch_dataloader(prepped_dir, batch_size=BATCH_SIZE)

    if get_train:
        dataloaders["train"] = process_split("train")
        print("--" * 20)

    if get_test:
        dataloaders["test"] = process_split("test")
        print("--" * 20)
        
    return dataloaders