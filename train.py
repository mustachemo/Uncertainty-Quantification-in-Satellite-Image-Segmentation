import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, LearningRateScheduler
import numpy as np
from pathlib import Path

from model.unet import build_unet_model
from model.bayesian_unet import build_bayesian_unet_model
from utils.visualize import visualize
from utils.logger_prep import get_logger
from utils.custom_funcs import dice_coefficient, combined_loss, combined_loss_bayesian_unet, uncertainty_aware_loss
from utils.directories_check import check_dirs, check_prepped_data
from configs import *

logger = get_logger(__name__)

def train_unet(train_dataset, test_dataset):
    if Path(f'checkpoints/unet_model_{DROPOUT_RATE}_{ACTIVATION_FUNC}.h5').exists():
        logger.info('Model already exists, exiting...')
        return

    tf.profiler.experimental.start(logdir='/logs/profiler/')
    logger.info('Model not found, creating and training...')
    model = build_unet_model(dropout_rate=DROPOUT_RATE)
    model.compile(optimizer='adam', loss=combined_loss, metrics=[dice_coefficient])

    # Callbacks
    # montior dice coefficient
    checkpoint = ModelCheckpoint(f'checkpoints/unet_model_{DROPOUT_RATE}_{ACTIVATION_FUNC}.keras', monitor='val_dice_coefficient', save_best_only=True, mode='max')
    tensorboard = TensorBoard(log_dir='logs')
    csv_logger = CSVLogger(f'logs/model_{DROPOUT_RATE}_{ACTIVATION_FUNC}_training.log')

    # Train the model
    model.fit(train_dataset, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(test_dataset), callbacks=[checkpoint, tensorboard, csv_logger], verbose=2)
    logger.info('Training complete for UNet model')
    tf.profiler.experimental.stop()

def train_bayesian_unet(train_dataset, test_dataset):
    if Path(f'checkpoints/bayesian_unet_model_{DROPOUT_RATE}_{ACTIVATION_FUNC}.h5').exists():
        logger.info('Model already exists, exiting...')
        pass

    logger.info('Model not found, creating and training...')
    model = build_bayesian_unet_model(dropout_rate=DROPOUT_RATE)
    model.compile(optimizer='adam', loss=combined_loss_bayesian_unet, metrics=['accuracy', dice_coefficient])

    # Callbacks
    checkpoint = ModelCheckpoint(f'checkpoints/bayesian_unet_model_{DROPOUT_RATE}_{ACTIVATION_FUNC}.h5', monitor='val_dice_coefficient', save_best_only=True, mode='max')
    tensorboard = TensorBoard(log_dir='logs')
    csv_logger = CSVLogger('logs/training.log')

    # Train the model
    model.fit(train_dataset, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(test_dataset), callbacks=[checkpoint, tensorboard, csv_logger])
    logger.info('Training complete for Bayesian UNet model')


if __name__ == '__main__':
    dataset = dict()

    check_dirs()
    dataset = check_prepped_data(get_train=True, get_test=True)

    gpu = tf.config.experimental.list_physical_devices('GPU')

    if not gpu:
        logger.warning('No GPU found, model may be slow or fail to train')
        
        # Number of threads to use for parallelizing the execution of an individual op.
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(4)
    else:
        logger.info(f'GPU found!')
        # Allow memory growth to avoid OOM errors
        for device in gpu:
            tf.config.experimental.set_memory_growth(device, True)

    
    # Check if the data is loaded correctly and normalized
    train_image_sample, train_mask_sample = dataset['train'].take(1)
    visualize(train_image_sample, train_mask_sample)
    # Check min and max values of the images and masks
    logger.info(f'Min and max values of images: {np.min(train_image_sample), np.max(train_image_sample)}')

    # visualize_train_sample(train_images[4], train_masks[4])

    # Train the UNet model
    train_unet(dataset['train'], dataset['test'])

    # Train the Bayesian UNet model
    #! Does not work currently due to memory issues
    # train_bayesian_unet(dataset['train'], dataset['test'])









    
