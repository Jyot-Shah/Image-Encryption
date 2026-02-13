"""
Attack Simulation and Robustness Testing
Implements Objective 4: Noise and occlusion attacks.
"""

import numpy as np
from typing import Dict, Tuple
import matplotlib.pyplot as plt
from metrics import calculate_psnr, calculate_ssim


def add_salt_pepper_noise(image: np.ndarray, salt_prob: float = 0.05, 
                          pepper_prob: float = 0.05) -> np.ndarray:
    """
    Add salt-and-pepper noise to encrypted image.
    
    Args:
        image: Encrypted image
        salt_prob: Probability of salt noise (white pixels)
        pepper_prob: Probability of pepper noise (black pixels)
    
    Returns:
        Noisy image
    """
    original_shape = image.shape
    flat = image.flatten().copy()
    
    # Total number of pixels
    total_pixels = flat.shape[0]
    
    # Add salt noise (255)
    num_salt = int(total_pixels * salt_prob)
    salt_indices = np.random.choice(total_pixels, num_salt, replace=False)
    flat[salt_indices] = 255
    
    # Add pepper noise (0)
    num_pepper = int(total_pixels * pepper_prob)
    pepper_indices = np.random.choice(total_pixels, num_pepper, replace=False)
    flat[pepper_indices] = 0
    
    return flat.reshape(original_shape).astype(np.uint8)


def occlusion_attack(image: np.ndarray, crop_percentage: int = 25) -> np.ndarray:
    """
    Simulate occlusion/cropping attack by zeroing a portion of the image.
    
    Args:
        image: Encrypted image (must be 2D or will be reshaped to 256x256)
        crop_percentage: Percentage of image to occlude (10, 25, 50, etc.)
    
    Returns:
        Image with occluded region
    """
    if len(image.shape) == 1:
        image = image.reshape(256, 256)
    
    occluded = image.copy()
    height, width = occluded.shape
    
    # Calculate crop dimensions (from top-left corner)
    crop_h = int(height * crop_percentage / 100)
    crop_w = int(width * crop_percentage / 100)
    
    # Zero out the region
    occluded[:crop_h, :crop_w] = 0
    
    return occluded


def test_salt_pepper_attack(original: np.ndarray, encrypted: np.ndarray,
                            decrypt_func, keys: Dict,
                            densities: list = [0.01, 0.05, 0.1]) -> Dict:
    """
    Test robustness against salt-and-pepper noise at multiple density levels.
    
    Args:
        original: Original image
        encrypted: Encrypted image
        decrypt_func: Decryption function
        keys: Dictionary with keys
        densities: List of noise density values (combined salt+pepper)
    
    Returns:
        Dictionary with results for each density level
    """
    results = {}
    
    for density in densities:
        # Split density between salt and pepper
        salt_prob = density / 2
        pepper_prob = density / 2
        
        # Add salt-and-pepper noise
        noisy_encrypted = add_salt_pepper_noise(encrypted, salt_prob, pepper_prob)
        
        # Attempt to decrypt
        try:
            decrypted = decrypt_func(noisy_encrypted, keys)
            
            # Measure quality
            psnr_val = calculate_psnr(original, decrypted)
            ssim_val = calculate_ssim(original, decrypted)
            
            results[f'density_{density}'] = {
                'psnr': psnr_val,
                'ssim': ssim_val,
                'noisy_encrypted': noisy_encrypted,
                'decrypted': decrypted
            }
        except Exception as e:
            results[f'density_{density}'] = {
                'error': str(e),
                'psnr': 0,
                'ssim': 0
            }
    
    return results


def test_occlusion_attack(original: np.ndarray, encrypted: np.ndarray,
                         decrypt_func, keys: Dict,
                         crop_percentages: list = [10, 25, 50]) -> Dict:
    """
    Test robustness against occlusion/cropping attacks.
    
    Args:
        original: Original image
        encrypted: Encrypted image
        decrypt_func: Decryption function
        keys: Dictionary with keys
        crop_percentages: List of crop percentages to test
    
    Returns:
        Dictionary with results for each crop percentage
    """
    results = {}
    
    for percentage in crop_percentages:
        # Apply occlusion attack
        occluded_encrypted = occlusion_attack(encrypted, percentage)
        
        # Attempt to decrypt
        try:
            decrypted = decrypt_func(occluded_encrypted.flatten(), keys)
            
            # Measure quality
            psnr_val = calculate_psnr(original, decrypted)
            ssim_val = calculate_ssim(original, decrypted)
            
            results[f'crop_{percentage}%'] = {
                'psnr': psnr_val,
                'ssim': ssim_val,
                'occluded_encrypted': occluded_encrypted,
                'decrypted': decrypted
            }
        except Exception as e:
            results[f'crop_{percentage}%'] = {
                'error': str(e),
                'psnr': 0,
                'ssim': 0
            }
    
    return results


def visualize_attack_results(original: np.ndarray, attack_results: Dict, 
                             attack_type: str, save_path: str = None):
    """
    Visualize attack results with before/after comparisons.
    
    Args:
        original: Original image
        attack_results: Results from attack test
        attack_type: Type of attack ('gaussian', 'salt_pepper', 'occlusion')
        save_path: Path to save visualization (optional)
    """
    n_tests = len(attack_results)
    fig, axes = plt.subplots(n_tests, 3, figsize=(12, 4 * n_tests))
    
    if n_tests == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (test_name, result) in enumerate(attack_results.items()):
        if 'error' in result:
            continue
        
        # Original
        axes[idx, 0].imshow(original, cmap='gray', vmin=0, vmax=255)
        axes[idx, 0].set_title('Original Image')
        axes[idx, 0].axis('off')
        
        # Attacked encrypted
        if 'noisy_encrypted' in result:
            attacked = result['noisy_encrypted']
        elif 'occluded_encrypted' in result:
            attacked = result['occluded_encrypted']
        else:
            continue
        
        if len(attacked.shape) == 1:
            attacked = attacked.reshape(256, 256)
        axes[idx, 1].imshow(attacked, cmap='gray', vmin=0, vmax=255)
        axes[idx, 1].set_title(f'Attacked Encrypted ({test_name})')
        axes[idx, 1].axis('off')
        
        # Decrypted after attack
        decrypted = result['decrypted']
        if len(decrypted.shape) == 1:
            decrypted = decrypted.reshape(256, 256)
        axes[idx, 2].imshow(decrypted, cmap='gray', vmin=0, vmax=255)
        axes[idx, 2].set_title(f'Decrypted (PSNR: {result["psnr"]:.2f} dB, SSIM: {result["ssim"]:.4f})')
        axes[idx, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Attack visualization saved to {save_path}")
    
    plt.close()


def run_all_attacks(original: np.ndarray, encrypted: np.ndarray,
                   decrypt_func, keys: Dict, save_dir: str = 'results/attack_analysis') -> Dict:
    """
    Run all attack tests and generate comprehensive report.
    
    Args:
        original: Original image
        encrypted: Encrypted image
        decrypt_func: Decryption function
        keys: Encryption keys
        save_dir: Directory to save results
    
    Returns:
        Dictionary with all attack results
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    all_results = {}
    
    # Salt-and-pepper attacks
    print("Testing salt-and-pepper noise attacks...")
    salt_pepper_results = test_salt_pepper_attack(original, encrypted, decrypt_func, keys)
    all_results['salt_pepper'] = salt_pepper_results
    visualize_attack_results(original, salt_pepper_results, 'salt_pepper',
                            os.path.join(save_dir, 'salt_pepper_attack.png'))
    
    # Occlusion attacks
    print("Testing occlusion attacks...")
    occlusion_results = test_occlusion_attack(original, encrypted, decrypt_func, keys)
    all_results['occlusion'] = occlusion_results
    visualize_attack_results(original, occlusion_results, 'occlusion',
                            os.path.join(save_dir, 'occlusion_attack.png'))
    
    return all_results
