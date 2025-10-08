from configs import BATCH_SIZE, MIXED_PRECISION
import matplotlib.pyplot as plt

def calculate_memory_usage(image, mask):
    bytes = 2 if MIXED_PRECISION else 4
    image_size = image.shape[0] * image.shape[1] * image.shape[2] * bytes
    mask_size = mask.shape[0] * mask.shape[1] * mask.shape[2] * bytes
    batch_size = BATCH_SIZE * (image_size + mask_size)
    return batch_size / 1024**2

def plot_img_and_mask(img, mask):
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i + 1})')
        ax[i + 1].imshow(mask == i)
    plt.xticks([]), plt.yticks([])
    plt.show()
