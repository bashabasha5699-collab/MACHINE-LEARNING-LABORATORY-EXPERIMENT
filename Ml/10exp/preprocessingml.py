"""
Image Preprocessing Module for ECG and Heart Rate Analysis
Handles loading, grayscale conversion, noise filtering, thresholding,
and morphological skeletonization to isolate the continuous cardiac waveform.
"""

import cv2
import numpy as np
from PIL import Image
import io

def load_image(image_input):
    """
    Load an image from various input types: filepath, bytes, PIL Image, or numpy array.
    Returns:
        np.ndarray: RGB image (H, W, 3) as uint8
    """
    if isinstance(image_input, str):
        img_bgr = cv2.imread(image_input)
        if img_bgr is None:
            raise ValueError(f"Could not open image from path: {image_input}")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, bytes):
        nparr = np.frombuffer(image_input, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, Image.Image):
        return np.array(image_input.convert("RGB"))
    elif hasattr(image_input, "read"):  # Streamlit UploadedFile or file-like object
        image_input.seek(0)
        file_bytes = image_input.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 2:
            return cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        return image_input
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

def to_grayscale(img_rgb):
    """Convert RGB image to 8-bit single-channel grayscale."""
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

def enhance_contrast(gray_img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Enhances the contrast between ECG signal traces and paper grid background.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray_img)

def denoise_image(gray_img, method="gaussian", kernel_size=3):
    """
    Apply noise filtering to suppress high-frequency speckle and scanner grain.
    """
    k = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
    if method == "gaussian":
        return cv2.GaussianBlur(gray_img, (k, k), 0)
    elif method == "median":
        return cv2.medianBlur(gray_img, k)
    elif method == "bilateral":
        return cv2.bilateralFilter(gray_img, d=9, sigmaColor=75, sigmaSpace=75)
    else:
        return gray_img

def binarize_image(gray_img, method="otsu", threshold_val=127, invert=True):
    """
    Convert grayscale image to binary (black and white).
    Otsu thresholding automatically finds the optimal threshold to separate
    dark ECG trace lines from bright paper background.
    """
    if method == "otsu":
        flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        _, binary = cv2.threshold(gray_img, 0, 255, flag + cv2.THRESH_OTSU)
    elif method == "adaptive":
        flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        binary = cv2.adaptiveThreshold(
            gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, flag, 11, 2
        )
    else:
        flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        _, binary = cv2.threshold(gray_img, threshold_val, 255, flag)
    return binary

def morphological_clean(binary_img, kernel_size=2):
    """
    Apply morphological opening and closing to remove isolated salt-and-pepper noise
    and connect minor gaps in the ECG waveform trace.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    # Opening removes small white noise
    opened = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
    # Closing bridges tiny discontinuities
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed

def extract_skeleton(binary_img):
    """
    Skeletonize / thin the binary image so the ECG line becomes 1 pixel wide.
    Uses morphological skeletonization for standard OpenCV compatibility.
    """
    skel = np.zeros(binary_img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    temp_img = binary_img.copy()

    done = False
    iterations = 0
    max_iters = 50

    while not done and iterations < max_iters:
        eroded = cv2.erode(temp_img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(temp_img, temp)
        skel = cv2.bitwise_or(skel, temp)
        temp_img = eroded.copy()

        zeros = binary_img.size - cv2.countNonZero(temp_img)
        if zeros == binary_img.size:
            done = True
        iterations += 1

    return skel if cv2.countNonZero(skel) > 0 else binary_img

def preprocess_pipeline(image_input, denoise_method="gaussian", kernel_size=3, binarize_method="otsu"):
    """
    Execute full image preprocessing pipeline and return each intermediate stage.
    Returns:
        dict: {
            'original_rgb': np.ndarray,
            'grayscale': np.ndarray,
            'clahe': np.ndarray,
            'denoised': np.ndarray,
            'binary': np.ndarray,
            'cleaned': np.ndarray,
            'skeleton': np.ndarray
        }
    """
    original = load_image(image_input)
    gray = to_grayscale(original)
    enhanced = enhance_contrast(gray)
    denoised = denoise_image(enhanced, method=denoise_method, kernel_size=kernel_size)
    binary = binarize_image(denoised, method=binarize_method, invert=True)
    cleaned = morphological_clean(binary, kernel_size=2)
    skeleton = extract_skeleton(cleaned)

    return {
        "original_rgb": original,
        "grayscale": gray,
        "clahe": enhanced,
        "denoised": denoised,
        "binary": binary,
        "cleaned": cleaned,
        "skeleton": skeleton,
    }
