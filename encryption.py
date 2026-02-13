"""
Level 1 (Permutation) and Level 2 (XOR Diffusion) Encryption
Implements confusion and diffusion stages of the three-level encryption system.
"""

import numpy as np
from typing import Tuple


def permute_pixels(image: np.ndarray, k1: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Level 1: Permutation (Confusion Stage)
    Randomly shuffles pixel positions using secret key k1.
    
    Args:
        image: 2D numpy array of shape (256, 256) with pixel values [0-255]
        k1: Secret key for permutation (integer seed)
    
    Returns:
        Tuple of (permuted_image, permutation_indices)
        - permuted_image: Flattened permuted image
        - permutation_indices: Array to reverse the permutation
    """
    # Flatten the image
    flat_image = image.flatten()
    total_pixels = flat_image.shape[0]  # Should be 65536 for 256x256
    
    # Generate deterministic permutation using k1 as seed
    np.random.seed(k1)
    perm_indices = np.random.permutation(total_pixels)
    
    # Apply permutation
    permuted_image = flat_image[perm_indices]
    
    return permuted_image, perm_indices


def xor_diffusion(image: np.ndarray, k2: int) -> np.ndarray:
    """
    Level 2: XOR-based Diffusion with 2-byte feedback chaining.
    Uses a keystream and 2 previous cipher bytes to maximize avalanche while staying invertible.
    Enhanced for higher NPCR.
    """
    rng = np.random.default_rng(k2)
    keystream = rng.integers(0, 256, size=image.shape[0], dtype=np.uint8)

    out = np.empty_like(image, dtype=np.uint8)
    prev1 = keystream[0]
    prev2 = keystream[0] if len(image) < 2 else keystream[1]
    
    for i, px in enumerate(image.astype(np.uint8)):
        # 2-byte feedback: XOR with keystream and both previous cipher bytes
        out[i] = np.uint8(px ^ keystream[i] ^ prev1 ^ prev2)
        prev2 = prev1
        prev1 = out[i]

    return out


def encrypt_level_1_2(image: np.ndarray, k1: int, k2: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combined Level 1 and Level 2 encryption.
    
    Args:
        image: 2D numpy array of shape (256, 256)
        k1: Secret key for permutation
        k2: Secret key for XOR diffusion
    
    Returns:
        Tuple of (diffused_image, permutation_indices)
    """
    # Level 1: Permutation
    permuted, perm_indices = permute_pixels(image, k1)
    
    # Level 2: XOR Diffusion
    diffused = xor_diffusion(permuted, k2)
    
    return diffused, perm_indices


def measure_entropy(image: np.ndarray) -> float:
    """
    Calculate Shannon entropy to measure confusion quality.
    
    Args:
        image: Numpy array of pixel values
    
    Returns:
        Entropy value (closer to 8.0 is better for 8-bit images)
    """
    # Get histogram
    hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
    
    # Normalize to get probabilities
    hist = hist / hist.sum()
    
    # Remove zero probabilities
    hist = hist[hist > 0]
    
    # Calculate entropy: H = -Σ(p * log2(p))
    entropy = -np.sum(hist * np.log2(hist))
    
    return entropy
