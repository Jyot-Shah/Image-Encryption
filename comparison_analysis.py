"""
Comparative Analysis for Objectives 3 and 5
Analyzes ANN architecture impact and compares with baseline methods.
"""

import numpy as np
import time
from typing import Dict, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, models
import os


def test_ann_architecture_variations(sample_images: list, k1: int, k2: int,
                                     save_dir: str = 'comparisons') -> pd.DataFrame:
    """
    Objective 3: Test different ANN architectures and measure impact on security.
    
    Tests variations in:
    - Hidden layer sizes (64, 128, 256, 512)
    - Activation functions (tanh, relu, sigmoid)
    - Training epochs (5, 10, 15, 20)
    
    Args:
        sample_images: List of sample images for training
        k1: Permutation key
        k2: Diffusion key
        save_dir: Directory to save results
    
    Returns:
        DataFrame with comparative results
    """
    from encryption import encrypt_level_1_2, measure_entropy
    from metrics import calculate_npcr, calculate_uaci
    
    results = []

    # Training data derived from deterministic substitution table
    from ann_model import prepare_training_data
    X_train, y_train, _, _ = prepare_training_data(k1, k2)
    test_img = sample_images[0]

    # Test different hidden layer sizes
    hidden_sizes = [64, 128, 256, 512]
    print("\nTesting different hidden layer sizes...")

    for hidden_size in hidden_sizes:
        print(f"  Testing hidden size: {hidden_size}")
        
        # Create custom ANN with this hidden size
        model = models.Sequential([
            layers.Dense(hidden_size, input_dim=256, activation='tanh'),
            layers.Dense(256, activation='softmax')
        ])
        model.compile(optimizer=keras.optimizers.Adam(0.001), loss='categorical_crossentropy')
        
        start_time = time.time()
        model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=0)
        train_time = time.time() - start_time
        
        # Test encryption on sample image
        encrypted, _ = encrypt_level_1_2(test_img, k1, k2)
        
        # Apply ANN substitution
        from ann_model import ann_substitute
        encrypted_ann = ann_substitute(encrypted, model)
        
        # Measure metrics
        entropy = measure_entropy(encrypted_ann)
        
        # For NPCR/UACI, encrypt with slightly different key
        encrypted2, _ = encrypt_level_1_2(test_img, k1 ^ 1, k2)
        encrypted_ann2 = ann_substitute(encrypted2, model)
        npcr = calculate_npcr(encrypted_ann, encrypted_ann2)
        uaci = calculate_uaci(encrypted_ann, encrypted_ann2)
        
        results.append({
            'config': f'hidden_{hidden_size}',
            'hidden_size': hidden_size,
            'activation': 'tanh',
            'epochs': 5,
            'train_time': train_time,
            'entropy': entropy,
            'npcr': npcr,
            'uaci': uaci
        })
    
    # Test different activation functions
    activations = ['tanh', 'relu', 'sigmoid']
    print("\nTesting different activation functions...")
    
    for activation in activations:
        print(f"  Testing activation: {activation}")
        
        model = models.Sequential([
            layers.Dense(256, input_dim=256, activation=activation),
            layers.Dense(256, activation='softmax')
        ])
        model.compile(optimizer=keras.optimizers.Adam(0.001), loss='categorical_crossentropy')
        
        start_time = time.time()
        model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=0)
        train_time = time.time() - start_time
        
        # Test encryption
        encrypted, _ = encrypt_level_1_2(test_img, k1, k2)
        encrypted_ann = ann_substitute(encrypted, model)
        
        entropy = measure_entropy(encrypted_ann)
        
        encrypted2, _ = encrypt_level_1_2(test_img, k1 ^ 1, k2)
        encrypted_ann2 = ann_substitute(encrypted2, model)
        npcr = calculate_npcr(encrypted_ann, encrypted_ann2)
        uaci = calculate_uaci(encrypted_ann, encrypted_ann2)
        
        results.append({
            'config': f'activation_{activation}',
            'hidden_size': 256,
            'activation': activation,
            'epochs': 10,
            'train_time': train_time,
            'entropy': entropy,
            'npcr': npcr,
            'uaci': uaci
        })
    
    # Test different training epochs
    epoch_counts = [5, 10, 15, 20]
    print("\nTesting different training epochs...")
    
    for epochs in epoch_counts:
        print(f"  Testing epochs: {epochs}")
        
        model = models.Sequential([
            layers.Dense(256, input_dim=256, activation='tanh'),
            layers.Dense(256, activation='softmax')
        ])
        model.compile(optimizer=keras.optimizers.Adam(0.001), loss='categorical_crossentropy')
        
        start_time = time.time()
        model.fit(X_train, y_train, epochs=epochs, batch_size=64, verbose=0)
        train_time = time.time() - start_time
        
        # Test encryption
        encrypted, _ = encrypt_level_1_2(test_img, k1, k2)
        encrypted_ann = ann_substitute(encrypted, model)
        
        entropy = measure_entropy(encrypted_ann)
        
        encrypted2, _ = encrypt_level_1_2(test_img, k1 ^ 1, k2)
        encrypted_ann2 = ann_substitute(encrypted2, model)
        npcr = calculate_npcr(encrypted_ann, encrypted_ann2)
        uaci = calculate_uaci(encrypted_ann, encrypted_ann2)
        
        results.append({
            'config': f'epochs_{epochs}',
            'hidden_size': 256,
            'activation': 'tanh',
            'epochs': epochs,
            'train_time': train_time,
            'entropy': entropy,
            'npcr': npcr,
            'uaci': uaci
        })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, 'ann_architecture_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nANN architecture comparison saved to {csv_path}")
    
    return df


def test_key_sensitivity(image: np.ndarray, k1: int, k2: int, forward_model,
                        inverse_model, perm_indices: np.ndarray,
                        substitution_table: np.ndarray = None,
                        inverse_table: np.ndarray = None) -> Dict:
    """
    Test sensitivity to single-bit key changes (Objective 3).
    
    Args:
        image: Original image
        k1: Original key 1
        k2: Original key 2
        forward_model: Trained forward ANN
        inverse_model: Trained inverse ANN
        perm_indices: Permutation indices
    
    Returns:
        Dictionary with key sensitivity metrics
    """
    from encryption import encrypt_level_1_2
    from ann_model import ann_substitute, ann_reverse
    from decryption import decrypt_level_2_1
    from metrics import calculate_npcr
    
    # Encrypt with correct keys
    diffused, _ = encrypt_level_1_2(image, k1, k2)
    encrypted = ann_substitute(diffused, forward_model, substitution_table=substitution_table)
    
    # Decrypt with correct keys
    recovered_ann = ann_reverse(encrypted, inverse_model, inverse_table=inverse_table,
                               substitution_table=substitution_table)
    correct_decrypted = decrypt_level_2_1(recovered_ann, k2, perm_indices)
    
    # Test with k1 changed by 1 bit
    k1_wrong = k1 ^ 1
    diffused_wrong, perm_indices_wrong = encrypt_level_1_2(image, k1_wrong, k2)
    encrypted_wrong_k1 = ann_substitute(diffused_wrong, forward_model, substitution_table=substitution_table)
    npcr_k1 = calculate_npcr(encrypted, encrypted_wrong_k1)
    
    # Test with k2 changed by 1 bit
    k2_wrong = k2 ^ 1
    diffused_wrong2, _ = encrypt_level_1_2(image, k1, k2_wrong)
    encrypted_wrong_k2 = ann_substitute(diffused_wrong2, forward_model, substitution_table=substitution_table)
    npcr_k2 = calculate_npcr(encrypted, encrypted_wrong_k2)
    
    return {
        'npcr_k1_1bit_change': npcr_k1,
        'npcr_k2_1bit_change': npcr_k2,
        'avg_key_sensitivity': (npcr_k1 + npcr_k2) / 2
    }


def compare_with_baselines(image: np.ndarray, save_dir: str = 'comparisons') -> pd.DataFrame:
    """
    Objective 5: Compare proposed method with baseline encryption methods.
    
    Compares:
    - Proposed ANN-based three-level encryption
    - Permutation-only encryption
    - XOR-only encryption
    - Combined permutation + XOR (without ANN)
    
    Args:
        image: Test image
        save_dir: Directory to save results
    
    Returns:
        DataFrame with comparative metrics
    """
    from encryption import permute_pixels, xor_diffusion, measure_entropy
    from metrics import calculate_npcr, calculate_uaci, correlation_coefficient
    
    results = []
    
    # Generate keys
    k1 = np.random.randint(0, 2**31)
    k2 = np.random.randint(0, 2**31)
    
    # 1. Permutation-only
    print("Testing permutation-only encryption...")
    start_time = time.time()
    perm_only, _ = permute_pixels(image, k1)
    perm_time = time.time() - start_time
    
    perm_only_reshaped = perm_only.reshape(256, 256)
    results.append({
        'method': 'Permutation Only',
        'encryption_time': perm_time,
        'entropy': measure_entropy(perm_only),
        'correlation_h': correlation_coefficient(perm_only_reshaped, 'horizontal'),
        'correlation_v': correlation_coefficient(perm_only_reshaped, 'vertical'),
        'correlation_d': correlation_coefficient(perm_only_reshaped, 'diagonal')
    })
    
    # 2. XOR-only
    print("Testing XOR-only encryption...")
    start_time = time.time()
    flat_img = image.flatten()
    xor_only = xor_diffusion(flat_img, k2)
    xor_time = time.time() - start_time
    
    xor_only_reshaped = xor_only.reshape(256, 256)
    results.append({
        'method': 'XOR Diffusion Only',
        'encryption_time': xor_time,
        'entropy': measure_entropy(xor_only),
        'correlation_h': correlation_coefficient(xor_only_reshaped, 'horizontal'),
        'correlation_v': correlation_coefficient(xor_only_reshaped, 'vertical'),
        'correlation_d': correlation_coefficient(xor_only_reshaped, 'diagonal')
    })
    
    # 3. Permutation + XOR (without ANN)
    print("Testing permutation + XOR encryption...")
    from encryption import encrypt_level_1_2
    start_time = time.time()
    perm_xor, _ = encrypt_level_1_2(image, k1, k2)
    perm_xor_time = time.time() - start_time
    
    perm_xor_reshaped = perm_xor.reshape(256, 256)
    results.append({
        'method': 'Permutation + XOR',
        'encryption_time': perm_xor_time,
        'entropy': measure_entropy(perm_xor),
        'correlation_h': correlation_coefficient(perm_xor_reshaped, 'horizontal'),
        'correlation_v': correlation_coefficient(perm_xor_reshaped, 'vertical'),
        'correlation_d': correlation_coefficient(perm_xor_reshaped, 'diagonal')
    })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, 'baseline_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nBaseline comparison saved to {csv_path}")
    
    return df


def visualize_architecture_comparison(df: pd.DataFrame, save_path: str):
    """
    Create visualizations for ANN architecture comparison.
    
    Args:
        df: DataFrame with comparison results
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Hidden size comparison
    hidden_df = df[df['config'].str.contains('hidden')]
    axes[0, 0].bar(hidden_df['hidden_size'].astype(str), hidden_df['entropy'])
    axes[0, 0].set_xlabel('Hidden Layer Size')
    axes[0, 0].set_ylabel('Entropy')
    axes[0, 0].set_title('Impact of Hidden Layer Size on Entropy')
    axes[0, 0].axhline(y=8.0, color='r', linestyle='--', label='Ideal (8.0)')
    axes[0, 0].legend()
    
    # Activation function comparison
    activation_df = df[df['config'].str.contains('activation')]
    x_pos = np.arange(len(activation_df))
    axes[0, 1].bar(x_pos, activation_df['npcr'])
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(activation_df['activation'])
    axes[0, 1].set_xlabel('Activation Function')
    axes[0, 1].set_ylabel('NPCR (%)')
    axes[0, 1].set_title('Impact of Activation Function on NPCR')
    axes[0, 1].axhline(y=99.6, color='r', linestyle='--', label='Ideal (99.6%)')
    axes[0, 1].legend()
    
    # Training epochs comparison
    epochs_df = df[df['config'].str.contains('epochs')]
    axes[1, 0].plot(epochs_df['epochs'], epochs_df['uaci'], marker='o')
    axes[1, 0].set_xlabel('Training Epochs')
    axes[1, 0].set_ylabel('UACI (%)')
    axes[1, 0].set_title('Impact of Training Epochs on UACI')
    axes[1, 0].axhline(y=33.4, color='r', linestyle='--', label='Ideal (33.4%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Training time comparison
    axes[1, 1].bar(epochs_df['epochs'].astype(str), epochs_df['train_time'])
    axes[1, 1].set_xlabel('Training Epochs')
    axes[1, 1].set_ylabel('Training Time (seconds)')
    axes[1, 1].set_title('Training Time vs Epochs')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Architecture comparison visualization saved to {save_path}")
    plt.close()


def visualize_baseline_comparison(df: pd.DataFrame, save_path: str):
    """
    Create radar chart comparing baseline methods.
    
    Args:
        df: DataFrame with baseline comparison results
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    
    # Metrics to compare (normalized to 0-1 scale)
    metrics = ['entropy', 'correlation_h', 'correlation_v', 'correlation_d']
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    # Plot each method
    for _, row in df.iterrows():
        values = [
            row['entropy'] / 8.0,  # Normalize to 0-1
            1 - abs(row['correlation_h']),  # Lower correlation is better
            1 - abs(row['correlation_v']),
            1 - abs(row['correlation_d'])
        ]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=row['method'])
        ax.fill(angles, values, alpha=0.15)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_title('Baseline Methods Comparison\n(Higher is Better)', size=14, weight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Baseline comparison visualization saved to {save_path}")
    plt.close()
