"""
BharatDoc AI - Image Preprocessing Module
Provides modular, evaluable image enhancements for noisy document scans.
All routines use Pillow and NumPy (pure CPU, zero heavy external binaries).
Preserves the raw image as the default baseline.
"""

from typing import Tuple
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np

PREPROCESSING_METHODS = {
    "none": "None (Raw Image - Benchmark Baseline)",
    "grayscale": "Grayscale Conversion",
    "contrast_boost": "Contrast Enhancement",
    "binarize_otsu": "Adaptive Binarization (Otsu Threshold)",
    "denoise": "Median Denoising",
    "sharpen": "Unsharp Masking (Edge Sharpening)",
}

class ImagePreprocessor:
    """Applies reproducible image preprocessing operations for OCR."""

    @staticmethod
    def to_grayscale(image: Image.Image) -> Image.Image:
        """Convert image to single-channel luminance."""
        return ImageOps.grayscale(image).convert("RGB")

    @staticmethod
    def enhance_contrast(image: Image.Image, factor: float = 1.6) -> Image.Image:
        """Increase contrast to make faded text discernible."""
        gray = ImageOps.grayscale(image)
        enhancer = ImageEnhance.Contrast(gray)
        enhanced = enhancer.enhance(factor)
        return enhanced.convert("RGB")

    @staticmethod
    def denoise(image: Image.Image, size: int = 3) -> Image.Image:
        """Apply median filter to suppress salt-and-pepper / Gaussian noise."""
        gray = ImageOps.grayscale(image)
        filtered = gray.filter(ImageFilter.MedianFilter(size=size))
        return filtered.convert("RGB")

    @staticmethod
    def sharpen(image: Image.Image) -> Image.Image:
        """Sharpen blurred text contours using unsharp mask."""
        gray = ImageOps.grayscale(image)
        sharpened = gray.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        return sharpened.convert("RGB")

    @staticmethod
    def binarize_otsu(image: Image.Image) -> Image.Image:
        """
        Compute Otsu's optimal threshold on grayscale histogram
        and binarize into high-contrast monochrome image.
        """
        gray = ImageOps.grayscale(image)
        np_img = np.array(gray)
        
        # Compute histogram
        hist, _ = np.histogram(np_img, bins=256, range=(0, 256))
        total = np_img.size
        current_max, threshold = 0.0, 128
        sum_total = np.dot(np.arange(256), hist)
        sum_back, weight_back = 0.0, 0

        for t in range(256):
            weight_back += hist[t]
            if weight_back == 0:
                continue
            weight_fore = total - weight_back
            if weight_fore == 0:
                break

            sum_back += t * hist[t]
            mean_back = sum_back / weight_back
            mean_fore = (sum_total - sum_back) / weight_fore

            between_variance = weight_back * weight_fore * ((mean_back - mean_fore) ** 2)
            if between_variance > current_max:
                current_max = between_variance
                threshold = t

        binarized = (np_img > threshold).astype(np.uint8) * 255
        return Image.fromarray(binarized).convert("RGB")

    @classmethod
    def apply_preprocessing(cls, image: Image.Image, method: str) -> Image.Image:
        """
        Dispatch selected preprocessing operation.
        Defaults to untouched raw image.
        """
        method = method.lower()
        if method == "none" or method not in PREPROCESSING_METHODS:
            return image
        elif method == "grayscale":
            return cls.to_grayscale(image)
        elif method == "contrast_boost":
            return cls.enhance_contrast(image)
        elif method == "binarize_otsu":
            return cls.binarize_otsu(image)
        elif method == "denoise":
            return cls.denoise(image)
        elif method == "sharpen":
            return cls.sharpen(image)
        return image
