from pathlib import Path
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from numpy import vstack
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.color import rgb2lab
import tensorflow

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "G_model" / "C4__256g_000040000.keras"
model = load_model(MODEL_PATH)
height, width = 256, 256

print("Model input shape:", model.input_shape)

def plot_images(src_img, gen_img, tar_img=None):
    if tar_img is not None:
        images = [src_img, gen_img, tar_img]
        titles = ['Source', 'Generated', 'Expected']
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    else:
        images = [src_img, gen_img]
        titles = ['Source', 'Generated']
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    for i, (img, title) in enumerate(zip(images, titles)):
        img = np.squeeze(img)
        img = (img + 1) / 2.0
        img = np.clip(img, 0, 1)
        
        if img.ndim == 2 or (img.ndim == 3 and img.shape[-1] == 1):
            axs[i].imshow(img, cmap='gray')
        else:
            axs[i].imshow(img)
        
        axs[i].axis('off')
        axs[i].set_title(title)
    plt.tight_layout()
    return fig

def preprocess_data(data):
    if isinstance(data, list):
        X1, X2 = data
        X1 = (X1 - 127.5) / 127.5
        X2 = (X2 - 127.5) / 127.5
        return [X1, X2]
    else:
        return (data - 127.5) / 127.5

def calculate_metrics(generated_image, target_image):
    generated_image = (generated_image + 1) / 2.0
    target_image = (target_image + 1) / 2.0
    
    if generated_image.ndim == 4 and generated_image.shape[-1] == 3:
        generated_image = np.mean(generated_image, axis=-1)
        target_image = np.mean(target_image, axis=-1)
    
    generated_image = np.squeeze(generated_image)
    target_image = np.squeeze(target_image)
    
    min_dim = min(generated_image.shape)
    win_size = min_dim if min_dim % 2 != 0 else min_dim - 1
    
    ssim_value = ssim(generated_image, target_image, win_size=win_size, data_range=1.0)
    psnr_value = psnr(target_image, generated_image, data_range=1.0)
    
    return ssim_value, psnr_value

def process_images(src_path, tar_path):
    src_image = load_img(src_path, target_size=(height, width), color_mode='rgb')
    src_image = img_to_array(src_image)
    src_image = np.expand_dims(src_image, axis=0)

    tar_image = load_img(tar_path, target_size=(height, width), color_mode='rgb')
    tar_image = img_to_array(tar_image)
    tar_image = np.expand_dims(tar_image, axis=0)

    src_image, tar_image = preprocess_data([src_image, tar_image])

    gen_image = model.predict(src_image)

    ssim_value, psnr_value = calculate_metrics(gen_image, tar_image)

    fig = plot_images(src_image[0], gen_image[0], tar_image[0])

    return fig, ssim_value, psnr_value, src_image[0], gen_image[0], tar_image[0]

def plot_histogram(image, ax, title):
    colors = ('r', 'g', 'b')
    for i, color in enumerate(colors):
        hist, bins = np.histogram(image[:, :, i].flatten(), bins=256, range=[0, 1])
        ax.plot(bins[:-1], hist, color=color, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel('Pixel Intensity')
    ax.set_ylabel('Count')

def plot_difference_map(original, generated):
    difference = np.abs(original - generated)
    
    # Enhance the difference for visibility
    difference = np.power(difference, 0.5)
    
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(difference, cmap='viridis')
    ax.set_title('Difference Map')
    fig.colorbar(im, ax=ax, label='Absolute Difference (sqrt-scaled)')
    return fig, np.mean(difference)

def colorize_image(image_file):
    # Load the image
    src_image = load_img(image_file, target_size=(height, width), color_mode='rgb')
    src_image = img_to_array(src_image)
    
    # Convert to grayscale
    src_image = np.mean(src_image, axis=-1, keepdims=True)
    
    # Repeat the grayscale channel to create a 3-channel image
    src_image = np.repeat(src_image, 3, axis=-1)
    
    # Normalize the image
    src_image = (src_image - 127.5) / 127.5
    
    # Add batch dimension
    src_image = np.expand_dims(src_image, axis=0)
    
    # Ensure the input shape is correct (batch, height, width, channels)
    expected_shape = (1, height, width, 3)
    if src_image.shape != expected_shape:
        raise ValueError(f"Expected input shape {expected_shape}, but got {src_image.shape}")
    
    # Generate the colorized image
    gen_image = model.predict(src_image)
    
    # Remove the batch dimension
    src_image = np.squeeze(src_image, axis=0)
    gen_image = np.squeeze(gen_image, axis=0)
    
    return src_image, gen_image

def plot_color_channels(image, title):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, channel in enumerate(['Red', 'Green', 'Blue']):
        axes[i].imshow(image[:,:,i], cmap='gray')
        axes[i].set_title(f'{channel} Channel')
        axes[i].axis('off')
    plt.suptitle(title)
    return fig

def plot_lab_channels(image, title):
    lab_image = rgb2lab(image)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    channels = ['L (Lightness)', 'a (Green-Red)', 'b (Blue-Yellow)']
    for i, channel in enumerate(channels):
        im = axes[i].imshow(lab_image[:,:,i], cmap='gray')
        axes[i].set_title(channel)
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i])
    plt.suptitle(title)
    return fig

# Print model summary for debugging
model.summary()
