# Opencv python Backend

This repository contains a Python-based backend for image processing tasks, using Flask and OpenCV to apply various transformations such as edge detection, Gaussian blur, image sharpening, and more. Future updates may include migration to alternative runtime backends for potential performance improvements. OpenCV is a widely used computer vision library in Python, enabling fast and efficient image computations Project Structure
Below is a recommended project structure for this repository:

## Project Structure

The folder structure of the project should be organized as follows:

```bash
opencv-python-backend/
├── backend.py
└── README.md

## Model Backend

Currently, the backend runs purely on Python with OpenCV. In the future, you may integrate additional optimization or inference frameworks to handle large-scale or more complex image processing tasks.

## Setup and Inference

To set up and run inference , follow these steps:

1. **Install UV :**
   ```bash
   pip install uv
   ```
2. **run uv**
   ```bash
   uv run backend.py   


## Operations:
* Edge Detection ("edge_detection")
 a Gaussian blur followed by the Canny edge detection algorithm.
* Gaussian Blur ("gaussian_blur")
*Blurs an image using a specified kernel size.
* Sharpen ("sharpen")
 Sharpens an image using a particular filter kernel.
* Grayscale ("grayscale")
Converts an image to grayscale.
* Median Blur ("median_blur")
Reduces noise using a median filter.
* Sobel Edge ("sobel_edge")
Detects edges using the Sobel operator in both horizontal and vertical directions.
