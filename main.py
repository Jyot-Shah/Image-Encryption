"""
Main Orchestration Script for ANN-Based Image Encryption System
Implements complete three-level encryption/decryption pipeline with evaluation.
"""

import numpy as np
import os
import time
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional
import csv

# Import custom modules
from encryption import permute_pixels, xor_diffusion, encrypt_level_1_2, measure_entropy
from decryption import reverse_xor_diffusion, inverse_permute, decrypt_level_2_1
from ann_model import (
    train_ann_pair,
    ann_substitute,
    ann_reverse,
    load_ann_models,
    create_ann_model,
    load_substitution_table,
    generate_substitution_tables,
)
from metrics import (evaluate_encryption_quality, calculate_entropy, calculate_npcr,
                    calculate_uaci, correlation_coefficient, calculate_psnr, calculate_ssim)
from attacks import run_all_attacks
from comparison_analysis import (test_ann_architecture_variations, test_key_sensitivity,
                                compare_with_baselines, visualize_architecture_comparison,
                                visualize_baseline_comparison)
from preprocess_sipi import preprocess_sipi_dataset


def load_and_preprocess_image(image_path: str) -> np.ndarray:
    """
    Load image and preprocess to 256x256 grayscale.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Preprocessed image as numpy array
    """
    # Load image
    img = Image.open(image_path)
    
    # Convert to grayscale
    img = img.convert('L')
    
    # Resize to 256x256
    img = img.resize((256, 256), Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    img_array = np.array(img, dtype=np.uint8)
    
    return img_array


def collect_image_paths(base_dir: str, max_images: Optional[int] = None) -> List[str]:
    """Recursively gather image paths with supported extensions."""
    if not os.path.exists(base_dir):
        return []

    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    paths: List[str] = []
    for root, _, files in os.walk(base_dir):
        for fname in files:
            if fname.lower().endswith(exts):
                paths.append(os.path.join(root, fname))

    paths.sort()
    if max_images is not None:
        return paths[:max_images]
    return paths


def generate_encryption_keys() -> Dict:
    """
    Generate random encryption keys.
    
    Returns:
        Dictionary with k1, k2 seeds
    """
    k1 = np.random.randint(0, 2**31)
    k2 = np.random.randint(0, 2**31)
    
    return {'k1': k1, 'k2': k2}


def save_keys(keys: Dict, save_path: str = 'saved_keys/encryption_keys.npz'):
    """
    Save encryption keys to file.
    
    Args:
        keys: Dictionary with keys
        save_path: Path to save keys
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, k1=keys['k1'], k2=keys['k2'])
    print(f"Keys saved to {save_path}")


def load_keys(load_path: str = 'saved_keys/encryption_keys.npz') -> Dict:
    """
    Load encryption keys from file.
    
    Args:
        load_path: Path to load keys from
    
    Returns:
        Dictionary with keys
    """
    data = np.load(load_path)
    return {'k1': int(data['k1']), 'k2': int(data['k2'])}


def encrypt_image_complete(image: np.ndarray, k1: int, k2: int, forward_model, substitution_table: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Complete three-level encryption pipeline.
    
    Args:
        image: Original image (256x256)
        k1: Permutation key
        k2: XOR diffusion key
        forward_model: Trained forward ANN model
    
    Returns:
        Tuple of (encrypted_image, permutation_indices, timing_info)
    """
    timing = {}
    
    # Level 1: Permutation
    start = time.time()
    permuted, perm_indices = permute_pixels(image, k1)
    timing['level_1'] = time.time() - start
    
    # Level 2: XOR Diffusion
    start = time.time()
    diffused = xor_diffusion(permuted, k2)
    timing['level_2'] = time.time() - start
    
    # Level 3: ANN Substitution
    start = time.time()
    encrypted = ann_substitute(diffused, forward_model, substitution_table=substitution_table)
    timing['level_3'] = time.time() - start
    
    timing['total'] = sum(timing.values())
    
    return encrypted, perm_indices, timing


def decrypt_image_complete(encrypted: np.ndarray, k2: int, perm_indices: np.ndarray,
                          inverse_model, inverse_table: np.ndarray = None,
                          substitution_table: np.ndarray = None) -> Tuple[np.ndarray, Dict]:
    """
    Complete three-level decryption pipeline.
    
    Args:
        encrypted: Encrypted image
        k2: XOR diffusion key
        perm_indices: Permutation indices from encryption
        inverse_model: Trained inverse ANN model
    
    Returns:
        Tuple of (decrypted_image, timing_info)
    """
    timing = {}
    
    # Reverse Level 3: Inverse ANN
    start = time.time()
    recovered_ann = ann_reverse(
        encrypted,
        inverse_model,
        inverse_table=inverse_table,
        substitution_table=substitution_table,
    )
    timing['level_3_inv'] = time.time() - start
    
    # Reverse Level 2: Reverse XOR
    start = time.time()
    recovered_xor = reverse_xor_diffusion(recovered_ann, k2)
    timing['level_2_inv'] = time.time() - start
    
    # Reverse Level 1: Inverse Permutation
    start = time.time()
    decrypted = inverse_permute(recovered_xor, perm_indices)
    timing['level_1_inv'] = time.time() - start
    
    timing['total'] = sum(timing.values())
    
    return decrypted, timing


def visualize_results(original: np.ndarray, encrypted: np.ndarray, decrypted: np.ndarray,
                     save_path: str = 'results/visualizations/encryption_demo.png'):
    """
    Visualize encryption and decryption results.
    
    Args:
        original: Original image
        encrypted: Encrypted image
        decrypted: Decrypted image
        save_path: Path to save visualization
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Reshape encrypted if needed
    encrypted_vis = encrypted.reshape(256, 256) if len(encrypted.shape) == 1 else encrypted
    
    # Original image
    axes[0, 0].imshow(original, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title('Original Image', fontsize=14, weight='bold')
    axes[0, 0].axis('off')
    
    # Encrypted image
    axes[0, 1].imshow(encrypted_vis, cmap='gray', vmin=0, vmax=255)
    axes[0, 1].set_title('Encrypted Image', fontsize=14, weight='bold')
    axes[0, 1].axis('off')
    
    # Decrypted image
    axes[0, 2].imshow(decrypted, cmap='gray', vmin=0, vmax=255)
    axes[0, 2].set_title('Decrypted Image', fontsize=14, weight='bold')
    axes[0, 2].axis('off')
    
    # Histograms
    axes[1, 0].hist(original.flatten(), bins=256, range=(0, 256), color='blue', alpha=0.7)
    axes[1, 0].set_title('Original Histogram', fontsize=12)
    axes[1, 0].set_xlabel('Pixel Value')
    axes[1, 0].set_ylabel('Frequency')
    
    axes[1, 1].hist(encrypted.flatten(), bins=256, range=(0, 256), color='red', alpha=0.7)
    axes[1, 1].set_title('Encrypted Histogram', fontsize=12)
    axes[1, 1].set_xlabel('Pixel Value')
    axes[1, 1].set_ylabel('Frequency')
    
    axes[1, 2].hist(decrypted.flatten(), bins=256, range=(0, 256), color='green', alpha=0.7)
    axes[1, 2].set_title('Decrypted Histogram', fontsize=12)
    axes[1, 2].set_xlabel('Pixel Value')
    axes[1, 2].set_ylabel('Frequency')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")
    plt.close()


def visualize_multiple_images(images: List[np.ndarray], image_names: List[str],
                               keys: Dict, forward_model, inverse_model,
                               substitution_table: np.ndarray, inverse_table: np.ndarray,
                               save_dir: str = 'results/visualizations'):
    """Visualize encryption/decryption for multiple images."""
    os.makedirs(save_dir, exist_ok=True)
    
    for idx, (image, img_name) in enumerate(zip(images, image_names), 1):
        print(f"  Visualizing image {idx}: {os.path.basename(img_name)}")
        
        encrypted, perm_indices, _ = encrypt_image_complete(
            image, keys['k1'], keys['k2'], forward_model, substitution_table=substitution_table
        )
        decrypted, _ = decrypt_image_complete(
            encrypted, keys['k2'], perm_indices, inverse_model,
            inverse_table=inverse_table, substitution_table=substitution_table
        )
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        encrypted_vis = encrypted.reshape(256, 256) if len(encrypted.shape) == 1 else encrypted
        
        axes[0, 0].imshow(image, cmap='gray', vmin=0, vmax=255)
        axes[0, 0].set_title(f'Original: {os.path.basename(img_name)}', fontsize=12, weight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(encrypted_vis, cmap='gray', vmin=0, vmax=255)
        axes[0, 1].set_title('Encrypted', fontsize=12, weight='bold')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(decrypted, cmap='gray', vmin=0, vmax=255)
        axes[0, 2].set_title('Decrypted', fontsize=12, weight='bold')
        axes[0, 2].axis('off')
        
        axes[1, 0].hist(image.flatten(), bins=256, range=(0, 256), color='blue', alpha=0.7)
        axes[1, 0].set_title('Original Histogram', fontsize=10)
        axes[1, 0].set_xlabel('Pixel Value')
        axes[1, 0].set_ylabel('Frequency')
        
        axes[1, 1].hist(encrypted.flatten(), bins=256, range=(0, 256), color='red', alpha=0.7)
        axes[1, 1].set_title('Encrypted Histogram', fontsize=10)
        axes[1, 1].set_xlabel('Pixel Value')
        axes[1, 1].set_ylabel('Frequency')
        
        axes[1, 2].hist(decrypted.flatten(), bins=256, range=(0, 256), color='green', alpha=0.7)
        axes[1, 2].set_title('Decrypted Histogram', fontsize=10)
        axes[1, 2].set_xlabel('Pixel Value')
        axes[1, 2].set_ylabel('Frequency')
        
        plt.tight_layout()
        safe_name = os.path.basename(img_name).replace('.', '_')
        save_path = os.path.join(save_dir, f'encryption_{idx}_{safe_name}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {save_path}")


def save_metrics_to_csv(metrics: Dict, filename: str = 'results/encryption_metrics.csv'):
    """
    Save metrics to CSV file.
    
    Args:
        metrics: Dictionary with metrics
        filename: Output CSV filename
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        for key, value in metrics.items():
            writer.writerow([key, value])
    
    print(f"Metrics saved to {filename}")


def save_metrics_table(metrics_list: List[Dict], filename: str = 'results/encryption_metrics_all.csv'):
    """Save metrics for multiple images into a single CSV table."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if not metrics_list:
        return

    # Collect headers from keys
    headers = ['image'] + [k for k in metrics_list[0] if k != 'image']

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in metrics_list:
            writer.writerow([row.get(h, '') for h in headers])

    print(f"Metrics table saved to {filename}")


def main():
    """
    Main execution function.
    """
    parser = argparse.ArgumentParser(description='ANN-Based Image Encryption System')
    parser.add_argument('--train', action='store_true', help='Train new ANN models')
    parser.add_argument('--test', action='store_true', help='Test encryption/decryption')
    parser.add_argument('--attacks', action='store_true', help='Run attack analysis')
    parser.add_argument('--compare', action='store_true', help='Run comparative analysis')
    parser.add_argument('--all', action='store_true', help='Run all analyses')
    parser.add_argument('--image', type=str, default=None, help='Path to test image')
    
    args = parser.parse_args()
    
    # Run all if specified
    if args.all:
        args.train = True
        args.test = True
        args.attacks = True
        args.compare = True
    
    data_root = 'data'
    processed_dir = os.path.join(data_root, 'processed')
    sipi_root = os.path.join(data_root, 'USC-SIPI Image Database')

    os.makedirs(data_root, exist_ok=True)

    # Load sample images
    print("\n" + "="*70)
    print("ANN-BASED THREE-LEVEL IMAGE ENCRYPTION SYSTEM")
    print("="*70 + "\n")

    sample_images: List[np.ndarray] = []

    # Prefer preprocessed SIPI set; fall back to downloader if empty
    image_paths = collect_image_paths(processed_dir)

    if len(image_paths) == 0 and os.path.exists(sipi_root):
        print("Processing USC-SIPI Image Database into 256x256 grayscale PNGs...")
        preprocess_sipi_dataset(sipi_root, processed_dir, size=256)
        image_paths = collect_image_paths(processed_dir)

    if len(image_paths) == 0:
        print("No images available. Please place USC-SIPI images under data/USC-SIPI Image Database/ and retry.")
        return

    max_train_images = min(20, len(image_paths))
    # Randomly select images for training
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    train_indices = rng.choice(len(image_paths), size=max_train_images, replace=False)
    train_paths = [image_paths[i] for i in sorted(train_indices)]
    
    for img_path in train_paths:
        img = load_and_preprocess_image(img_path)
        sample_images.append(img)
        rel_path = os.path.relpath(img_path, data_root)
        print(f"Loaded: {rel_path} -> {img.shape}")
    
    # Generate encryption keys
    keys = generate_encryption_keys()
    print(f"\nGenerated keys: k1={keys['k1']}, k2={keys['k2']}")
    save_keys(keys)
    
    substitution_table = load_substitution_table('saved_keys')

    # Train ANN models
    if args.train or not os.path.exists('saved_keys/forward_ann.h5'):
        print("\n" + "-"*70)
        print("TRAINING ANN MODELS (Level 3)")
        print("-"*70)
        forward_model, inverse_model = train_ann_pair(
            sample_images, keys['k1'], keys['k2'],
            epochs=50, save_dir='saved_keys'
        )
        substitution_table = load_substitution_table('saved_keys')
    else:
        print("\nLoading pre-trained ANN models...")
        forward_model, inverse_model = load_ann_models('saved_keys')
        substitution_table = load_substitution_table('saved_keys')
        print("Models loaded successfully.")

    if substitution_table is None:
        substitution_table, _ = generate_substitution_tables(keys['k1'], keys['k2'])

    inverse_table = np.argsort(substitution_table).astype(np.uint8)
    
    # Test encryption/decryption
    if args.test or args.all or (not args.attacks and not args.compare):
        print("\n" + "-"*70)
        print("TESTING ENCRYPTION/DECRYPTION PIPELINE")
        print("-"*70)

        # Evaluate and visualize images
        print("\nGenerating visualizations and metrics for multiple images...")
        all_paths = collect_image_paths(processed_dir)
        metrics_rows: List[Dict] = []

        for idx, img_path in enumerate(all_paths):
            img = load_and_preprocess_image(img_path)
            rel_path = os.path.relpath(img_path, data_root)
            print(f"  [{idx+1}] Processing: {rel_path}")

            # Encrypt
            encrypted, perm_indices, enc_timing = encrypt_image_complete(
                img, keys['k1'], keys['k2'], forward_model, substitution_table=substitution_table
            )

            # Decrypt
            decrypted, dec_timing = decrypt_image_complete(
                encrypted, keys['k2'], perm_indices, inverse_model, inverse_table=inverse_table,
                substitution_table=substitution_table
            )

            # Metrics
            metrics = evaluate_encryption_quality(img, encrypted, decrypted)
            encrypted2, _, _ = encrypt_image_complete(
                img, keys['k1'] ^ 1, keys['k2'], forward_model, substitution_table=substitution_table
            )
            npcr = calculate_npcr(encrypted, encrypted2)
            uaci = calculate_uaci(encrypted, encrypted2)

            metrics.update({
                'image': rel_path,
                'npcr': npcr,
                'uaci': uaci,
                'encryption_time': enc_timing['total'],
                'decryption_time': dec_timing['total'],
            })
            metrics_rows.append(metrics)

        # Select best 5 by encrypted entropy then NPCR
        metrics_rows_sorted = sorted(
            metrics_rows,
            key=lambda m: (m.get('encrypted_entropy', 0), m.get('npcr', 0)),
            reverse=True
        )
        top5 = metrics_rows_sorted[:5]
        save_metrics_table(top5, filename='results/encryption_metrics_all.csv')

        # Visualize the top-performing images listed in encryption_metrics_all.csv
        best_viz_dir = os.path.join('results', 'visualizations')
        os.makedirs(best_viz_dir, exist_ok=True)

        for rank, row in enumerate(top5, 1):
            rel_path = row.get('image')
            if not rel_path:
                continue

            full_path = os.path.join(data_root, os.path.normpath(rel_path))
            if not os.path.exists(full_path):
                print(f"  Skipping visualization (missing file): {full_path}")
                continue

            img = load_and_preprocess_image(full_path)
            encrypted, perm_indices, _ = encrypt_image_complete(
                img, keys['k1'], keys['k2'], forward_model, substitution_table=substitution_table
            )
            decrypted, _ = decrypt_image_complete(
                encrypted, keys['k2'], perm_indices, inverse_model,
                inverse_table=inverse_table, substitution_table=substitution_table
            )

            safe_name = os.path.basename(rel_path).replace('.', '_')
            vis_path = os.path.join(best_viz_dir, f'best_{rank}_{safe_name}.png')
            visualize_results(img, encrypted, decrypted, save_path=vis_path)

            encrypted_vis = encrypted.reshape(256, 256) if len(encrypted.shape) == 1 else encrypted
            Image.fromarray(encrypted_vis).save(os.path.join(best_viz_dir, f'best_{rank}_{safe_name}_encrypted.png'))
            Image.fromarray(decrypted).save(os.path.join(best_viz_dir, f'best_{rank}_{safe_name}_decrypted.png'))
            print(f"    Best visualization saved: {vis_path}")
    
    # Run attack analysis
    if args.attacks or args.all:
        print("\n" + "-"*70)
        print("ATTACK ROBUSTNESS ANALYSIS")
        print("-"*70)
        
        test_image = sample_images[0]
        encrypted, perm_indices, _ = encrypt_image_complete(
            test_image, keys['k1'], keys['k2'], forward_model, substitution_table=substitution_table
        )
        
        # Create decrypt function wrapper
        def decrypt_wrapper(enc_img, key_dict):
            decrypted, _ = decrypt_image_complete(
                enc_img, key_dict['k2'], key_dict['perm_indices'], key_dict['inverse_model'],
                inverse_table=key_dict['inverse_table'], substitution_table=key_dict['substitution_table']
            )
            return decrypted
        
        attack_keys = {
            'k2': keys['k2'],
            'perm_indices': perm_indices,
            'inverse_model': inverse_model,
            'inverse_table': inverse_table,
            'substitution_table': substitution_table,
        }
        
        attack_results = run_all_attacks(
            test_image, encrypted, decrypt_wrapper, attack_keys,
            save_dir='results/attack_analysis'
        )
        
        print("\nAttack analysis complete. Results saved to results/attack_analysis/")
    
    # Run comparative analysis
    if args.compare or args.all:
        print("\n" + "-"*70)
        print("COMPARATIVE ANALYSIS")
        print("-"*70)
        
        # ANN Architecture Analysis
        print("\nAnalyzing ANN architecture impact...")
        arch_df = test_ann_architecture_variations(
            sample_images, keys['k1'], keys['k2'], save_dir='comparisons'
        )
        visualize_architecture_comparison(
            arch_df, 'comparisons/ann_architecture_comparison.png'
        )
        
        # Baseline Comparison
        print("\nComparing with baseline methods...")
        baseline_df = compare_with_baselines(
            sample_images[0], save_dir='comparisons'
        )
        visualize_baseline_comparison(
            baseline_df, 'comparisons/baseline_comparison.png'
        )
        
        print("\nComparative analysis complete. Results saved to comparisons/")
    
    print("\n" + "="*70)
    print("ALL ANALYSES COMPLETE")
    print("="*70)
    print("\nResults Summary:")
    print("  - Encryption metrics: results/encryption_metrics.csv")
    print("  - Visualizations: results/visualizations/")
    print("  - Attack analysis: results/attack_analysis/")
    print("  - Comparisons: comparisons/")
    print("  - Saved keys: saved_keys/")
    print("\n")


if __name__ == '__main__':
    main()
