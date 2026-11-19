"""Pure NumPy math behind the MRI Fourier reconstruction exhibit.

This is a teaching model: FFT of a *finished* 2D MRI image, not acquisition
of raw magnetic-resonance signals from an actual scanner.
"""
from __future__ import annotations

import numpy as np


def to_grayscale_array(pil_image, size: int = 256) -> np.ndarray:
    """Load an image into a square real-valued 2D matrix in [0, 1]."""
    from PIL import Image, ImageOps

    image = ImageOps.fit(pil_image.convert("L"), (size, size), method=Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float64) / 255.0


def fft_image(image: np.ndarray) -> np.ndarray:
    """Centered 2D discrete Fourier transform of a real image matrix."""
    return np.fft.fftshift(np.fft.fft2(image))


def frequency_radius(shape: tuple[int, int]) -> np.ndarray:
    """Radial frequency coordinate, normalized so corners are about 1."""
    rows, cols = shape
    y = (np.arange(rows) - rows // 2) / (rows / 2)
    x = (np.arange(cols) - cols // 2) / (cols / 2)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    return np.sqrt(xx * xx + yy * yy) / np.sqrt(2.0)


def frequency_mask(
    shape: tuple[int, int],
    remove_high_pct: int = 0,
    remove_low_pct: int = 0,
) -> np.ndarray:
    """Remove the outer high-frequency band and/or inner low-frequency disk.

    Percentages refer to *radial distance* in the centered k-space square;
    they do NOT refer to the percentage of coefficients removed.
    """
    if not (0 <= remove_high_pct <= 100 and 0 <= remove_low_pct <= 100):
        raise ValueError("Frequency-cut percentages must lie between 0 and 100")
    r = frequency_radius(shape)
    high_limit = 1.0 - remove_high_pct / 100.0
    low_limit = remove_low_pct / 100.0
    return (r <= high_limit + 1e-12) & (r >= low_limit - 1e-12)


def reconstruct(image: np.ndarray, remove_high_pct: int = 0, remove_low_pct: int = 0):
    """Compute the forward transform, k-space mask and magnitude reconstruction."""
    original_k = fft_image(image)
    mask = frequency_mask(image.shape, remove_high_pct, remove_low_pct)
    observed_k = original_k * mask
    inverse = np.fft.ifft2(np.fft.ifftshift(observed_k))
    reconstructed = np.abs(inverse)  # common magnitude display convention
    return original_k, mask, observed_k, reconstructed, inverse


def kspace_rgb(kspace: np.ndarray, reference_kspace: np.ndarray) -> np.ndarray:
    """Log-scale k-space magnitude, mapped into a navy-to-cyan image."""
    level = np.log1p(np.abs(kspace))
    ref = np.log1p(np.abs(reference_kspace))
    scale = float(ref.max()) or 1.0
    strength = np.clip(level / scale, 0.0, 1.0) ** 0.73
    palette = np.array([6.0, 16.0, 29.0])
    glow = np.array([90.0, 229.0, 246.0])
    rgb = palette[None, None, :] + strength[..., None] * (glow - palette)[None, None, :]
    return np.uint8(np.clip(rgb, 0, 255))


def grayscale_rgb(image: np.ndarray) -> np.ndarray:
    """A view suitable for Streamlit, with fixed rather than auto contrast."""
    x = np.uint8(np.clip(image, 0, 1) * 255)
    return np.repeat(x[:, :, None], 3, axis=2)


def reconstruction_metrics(image: np.ndarray, recon: np.ndarray, mask: np.ndarray) -> dict:
    error = recon - image
    return {
        "kept_pct": 100.0 * float(mask.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "max_error": float(np.max(np.abs(error))),
        "all_data": bool(np.all(mask)),
    }
