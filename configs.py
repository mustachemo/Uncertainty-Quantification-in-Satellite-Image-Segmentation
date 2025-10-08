# Data configuration
X_DIMENSION = 256
Y_DIMENSION = 256

# Directory paths
TRAIN_IMAGES_DIR = "./data/images/train"
TRAIN_MASKS_DIR = "./data/masks/train"
TEST_IMAGES_DIR = "./data/images/val"
TEST_MASKS_DIR = "./data/masks/val"

# Prepped directories for PyTorch tensors
PREPPED_TRAIN_DATASET_DIR = "prepped_data/train"
PREPPED_TEST_DATASET_DIR = "prepped_data/test"

# Model paramaeters/hyperparameters
DROPOUT_RATE = 0.35
EPOCHS = 5
N_CHANNELS = 3 # RGB
N_CLASSES = 1 # Binary segmentation

# Data parameters
BATCH_SIZE = 16 # Adjusted for typical PyTorch training
MIXED_PRECISION = False

# Pick from :
# 'relu'      : ReLU is the most commonly used activation function in deep learning. It helps in mitigating the vanishing gradient problem and is computationally efficient
# 'elu'       : ELU can result in faster and more accurate learning by having negative values which push mean activations closer to zero.
# 'swish'     : Swish tends to outperform ReLU in deeper networks by avoiding the dead neuron problem and providing a smooth activation function.
# 'gelu'      : GELU smooths the output curve and tends to perform better on tasks involving natural language processing, and it may offer benefits in image tasks as well.
# 'leaky_relu': Leaky ReLU allows a small gradient when the unit is not active, which helps in avoiding dead neurons.
ACTIVATION_FUNC = "relu"


# Uncertainty quantification testing
NUM_SAMPLES_MC_DROPOUT_PREDICTION = 10
GRID_ITERATIONS = 5