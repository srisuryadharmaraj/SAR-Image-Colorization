# SAR Image Colorization using Pix2Pix GAN

An AI-powered satellite image colorization system that transforms grayscale Synthetic Aperture Radar (SAR) imagery into optical-style RGB images using a trained Pix2Pix GAN.

## Features

- SAR grayscale image input
- Optical target image comparison
- Pix2Pix GAN-based colorization
- SSIM quality evaluation
- PSNR quality evaluation
- RGB histogram analysis
- Difference map visualization
- RGB channel analysis
- LAB color-space analysis
- Interactive image comparison
- Streamlit-based dashboard

## Technologies Used

- Python
- TensorFlow 2.15
- Keras 2.15
- Streamlit
- NumPy
- Matplotlib
- Scikit-image
- Pillow

## Project Structure

```text
SAR/
├── Home.py
├── image_colorization.py
├── pix2pix_GAN.py
├── pix2pix_512.py
├── data_prep.py
├── requirements.txt
├── .gitignore
└── G_model/C4__256g_000040000.keras