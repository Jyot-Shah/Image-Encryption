"""
Security and Image Quality Metrics
Implements all evaluation metrics for Objective 1.
"""

import numpy as np
from scipy import stats
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from typing import Dict, Tuple


def calculate_entropy(image: np.ndarray) -> float:
    """
    Calculate Shannon entropy of an image.
    Ideal value for encrypted image: ~8.0 (maximum for 8-bit images)
    
    Args:
        image: 2D or 1D numpy array
    
    Returns:
        Entropy value in bits
    """
    # Flatten if 2D
    flat = image.flatten()
    
    # Get histogram
    hist, _ = np.histogram(flat, bins=256, range=(0, 256))
    
    # Normalize to probabilities
    hist = hist / hist.sum()
    
    # Remove zeros to avoid log(0)
    hist = hist[hist > 0]
    
    # Shannon entropy: H = -Σ(p * log2(p))
    entropy = -np.sum(hist * np.log2(hist))
    
    return entropy


def calculate_npcr(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Calculate Number of Pixel Change Rate (NPCR).
    Measures percentage of different pixels between two images.
    Ideal value: ~99.6% for good encryption
    
    Args:
        image1: First image
        image2: Second image (encrypted with slightly different key)
    
    Returns:
        NPCR value as percentage
    """
    # Flatten both images
    flat1 = image1.flatten()
    flat2 = image2.flatten()
    
    # Count different pixels
    different = np.sum(flat1 != flat2)
    total = flat1.shape[0]
    
    # Calculate NPCR as percentage
    npcr = (different / total) * 100.0
    
    return npcr


def calculate_uaci(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Calculate Unified Average Changing Intensity (UACI).
    Measures average intensity difference between two images.
    Ideal value: ~33.4% for good encryption
    
    Args:
        image1: First image
        image2: Second image (encrypted with slightly different key)
    
    Returns:
        UACI value as percentage
    """
    # Flatten both images
    flat1 = image1.flatten().astype(np.float64)
    flat2 = image2.flatten().astype(np.float64)
    
    # Calculate absolute differences
    diff = np.abs(flat1 - flat2)
    
    # Calculate UACI
    uaci = (np.sum(diff) / (flat1.shape[0] * 255.0)) * 100.0
    
    return uaci


def correlation_coefficient(image: np.ndarray, direction: str = 'horizontal') -> float:
    """
    Calculate correlation coefficient between adjacent pixels.
    Measures pixel correlation in specified direction.
    Ideal value for encrypted image: ~0.0 (no correlation)
    
    Args:
        image: 2D numpy array
        direction: 'horizontal', 'vertical', or 'diagonal'
    
    Returns:
        Correlation coefficient (-1 to 1)
    """
    if len(image.shape) == 1:
        # Reshape if flattened
        image = image.reshape(256, 256)
    
    if direction == 'horizontal':
        # Compare each pixel with its right neighbor
        x = image[:, :-1].flatten()
        y = image[:, 1:].flatten()
    elif direction == 'vertical':
        # Compare each pixel with its bottom neighbor
        x = image[:-1, :].flatten()
        y = image[1:, :].flatten()
    elif direction == 'diagonal':
        # Compare each pixel with its diagonal neighbor
        x = image[:-1, :-1].flatten()
        y = image[1:, 1:].flatten()
    else:
        raise ValueError("Direction must be 'horizontal', 'vertical', or 'diagonal'")
    
    # Calculate Pearson correlation coefficient
    if len(x) == 0:
        return 0.0
    
    correlation, _ = stats.pearsonr(x, y)
    
    return correlation


def calculate_psnr(original: np.ndarray, recovered: np.ndarray) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).
    Measures quality of decryption (how close recovered image is to original).
    Ideal value: >40 dB indicates excellent recovery
    
    Args:
        original: Original image
        recovered: Decrypted image
    
    Returns:
        PSNR value in dB
    """
    # Ensure both images have same shape
    if original.shape != recovered.shape:
        raise ValueError("Images must have the same dimensions")
    
    # Use skimage's PSNR implementation
    psnr_value = psnr(original, recovered, data_range=255)
    
    return psnr_value


def calculate_ssim(original: np.ndarray, recovered: np.ndarray) -> float:
    """
    Calculate Structural Similarity Index (SSIM).
    Measures structural similarity between original and recovered images.
    Ideal value: >0.99 indicates excellent recovery
    
    Args:
        original: Original image
        recovered: Decrypted image
    
    Returns:
        SSIM value (0 to 1)
    """
    # Ensure both images have same shape
    if original.shape != recovered.shape:
        raise ValueError("Images must have the same dimensions")
    
    # Reshape if flattened
    if len(original.shape) == 1:
        original = original.reshape(256, 256)
    if len(recovered.shape) == 1:
        recovered = recovered.reshape(256, 256)
    
    # Use skimage's SSIM implementation
    ssim_value = ssim(original, recovered, data_range=255)
    
    return ssim_value


def histogram_uniformity_test(image: np.ndarray) -> Dict[str, float]:
    """
    Test histogram uniformity of encrypted image.
    
    Args:
        image: Encrypted image
    
    Returns:
        Dictionary with chi-square statistic and p-value
    """
    # Get histogram
    hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
    
    # Expected uniform distribution
    expected = np.full(256, image.size / 256)
    
    # Chi-square test
    chi_square = np.sum((hist - expected) ** 2 / expected)
    
    # Degrees of freedom
    dof = 255
    
    # P-value (using chi-square distribution)
    p_value = 1 - stats.chi2.cdf(chi_square, dof)
    
    return {
        'chi_square': chi_square,
        'p_value': p_value,
        'is_uniform': p_value > 0.05  # Typically use 0.05 significance level
    }


def evaluate_encryption_quality(original: np.ndarray, encrypted: np.ndarray, 
                                decrypted: np.ndarray) -> Dict[str, any]:
    """
    Comprehensive evaluation of encryption quality.
    Calculates all metrics for Objective 1.
    
    Args:
        original: Original image
        encrypted: Encrypted image
        decrypted: Decrypted image
    
    Returns:
        Dictionary with all metric results
    """
    results = {}
    
    # Entropy
    results['original_entropy'] = calculate_entropy(original)
    results['encrypted_entropy'] = calculate_entropy(encrypted)
    
    # Correlation coefficients (original)
    results['original_corr_h'] = correlation_coefficient(original, 'horizontal')
    results['original_corr_v'] = correlation_coefficient(original, 'vertical')
    results['original_corr_d'] = correlation_coefficient(original, 'diagonal')
    
    # Correlation coefficients (encrypted)
    if len(encrypted.shape) == 1:
        encrypted_reshaped = encrypted.reshape(256, 256)
    else:
        encrypted_reshaped = encrypted
    results['encrypted_corr_h'] = correlation_coefficient(encrypted_reshaped, 'horizontal')
    results['encrypted_corr_v'] = correlation_coefficient(encrypted_reshaped, 'vertical')
    results['encrypted_corr_d'] = correlation_coefficient(encrypted_reshaped, 'diagonal')
    
    # Histogram uniformity
    uniformity = histogram_uniformity_test(encrypted)
    results['histogram_chi_square'] = uniformity['chi_square']
    results['histogram_uniform'] = uniformity['is_uniform']
    
    # Decryption quality
    if decrypted is not None:
        results['psnr'] = calculate_psnr(original, decrypted)
        results['ssim'] = calculate_ssim(original, decrypted)
    
    return results


def key_sensitivity_analysis(original: np.ndarray, k1: int, k2: int, 
                             encrypt_func, decrypt_func) -> Dict[str, float]:
    """
    Test sensitivity to key changes.
    
    Args:
        original: Original image
        k1: Original key 1
        k2: Original key 2
        encrypt_func: Encryption function
        decrypt_func: Decryption function
    
    Returns:
        Dictionary with sensitivity metrics
    """
    # Encrypt with original keys
    encrypted_correct, perm_indices = encrypt_func(original, k1, k2)
    
    # Try decrypting with slightly different k1 (flip one bit)
    k1_wrong = k1 ^ 1  # Flip least significant bit
    decrypted_wrong_k1 = decrypt_func(encrypted_correct, k1_wrong, k2, perm_indices)
    
    # Try decrypting with slightly different k2
    k2_wrong = k2 ^ 1
    decrypted_wrong_k2 = decrypt_func(encrypted_correct, k1, k2_wrong, perm_indices)
    
    # Calculate NPCR for wrong keys
    results = {
        'npcr_wrong_k1': calculate_npcr(original, decrypted_wrong_k1),
        'npcr_wrong_k2': calculate_npcr(original, decrypted_wrong_k2)
    }
    
    return results
