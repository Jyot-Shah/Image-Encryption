"""
Decryption Module - Reverses all three levels of encryption
Implements inverse operations for permutation, XOR diffusion, and ANN substitution.
"""

import numpy as np


def reverse_xor_diffusion(image: np.ndarray, k2: int) -> np.ndarray:
    """
    Reverse Level 2 diffusion with 2-byte feedback chaining.
    Inverts xor_diffusion by walking forward with stored cipher bytes.
    """
    rng = np.random.default_rng(k2)
    keystream = rng.integers(0, 256, size=image.shape[0], dtype=np.uint8)

    out = np.empty_like(image, dtype=np.uint8)
    prev1 = keystream[0]
    prev2 = keystream[0] if len(image) < 2 else keystream[1]
    
    for i, cx in enumerate(image.astype(np.uint8)):
        # Reverse with 2-byte feedback
        out[i] = np.uint8(cx ^ keystream[i] ^ prev1 ^ prev2)
        prev2 = prev1
        prev1 = cx  # use cipher byte for chaining

    return out


def inverse_permute(image: np.ndarray, perm_indices: np.ndarray, 
                   original_shape: tuple = (256, 256)) -> np.ndarray:
    """
    Reverse Level 1: Inverse Permutation
    Restores original pixel positions using stored permutation indices.
    
    Args:
        image: 1D numpy array (permuted pixels)
        perm_indices: Permutation indices used during encryption
        original_shape: Original 2D shape to restore (default 256x256)
    
    Returns:
        Restored 2D image array
    """
    # Create inverse permutation
    inv_indices = np.argsort(perm_indices)
    
    # Apply inverse permutation
    restored_flat = image[inv_indices]
    
    # Reshape to original dimensions
    restored_image = restored_flat.reshape(original_shape)
    
    return restored_image


def decrypt_level_2_1(image: np.ndarray, k2: int, perm_indices: np.ndarray) -> np.ndarray:
    """
    Combined reverse of Level 2 and Level 1.
    
    Args:
        image: 1D numpy array (after diffusion and permutation)
        k2: Secret key for XOR diffusion
        perm_indices: Permutation indices from encryption
    
    Returns:
        Decrypted 2D image array
    """
    # Reverse Level 2: XOR Diffusion
    permuted = reverse_xor_diffusion(image, k2)
    
    # Reverse Level 1: Inverse Permutation
    decrypted = inverse_permute(permuted, perm_indices)
    
    return decrypted
