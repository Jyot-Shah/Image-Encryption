"""
Level 3: ANN-Based Encryption (Substitution Layer)
Implements shallow feedforward neural network (256→512→256) as cryptographic key.
The trained ANN weights act as the encryption key for nonlinear pixel substitution.
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from typing import Tuple, Optional
import os


def create_ann_model() -> keras.Model:
    """
    Create shallow feedforward ANN architecture.
    
    Architecture:
        - Input layer: 256 neurons (one-hot for pixel value 0-255)
        - Hidden layer: 512 neurons with tanh activation (enhanced nonlinearity)
        - Output layer: 256 neurons with softmax activation
    
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        layers.Dense(
            512,
            input_dim=256,
            activation='tanh',
            kernel_regularizer=regularizers.l2(1e-5),
            name='hidden_layer'
        ),
        layers.Dense(
            256,
            activation='softmax',
            name='output_layer'
        )
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def generate_substitution_tables(k1: int, k2: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Derive deterministic substitution and inverse tables from keys.
    The table is a permutation of 0..255; inverse is its inverse permutation.
    """
    seed = (int(k1) ^ int(k2)) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    forward_table = np.arange(256, dtype=np.uint8)
    rng.shuffle(forward_table)
    inverse_table = np.argsort(forward_table).astype(np.uint8)
    return forward_table, inverse_table


def prepare_training_data(k1: int, k2: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build deterministic one-hot→one-hot mappings from key-derived substitution tables.

    Returns X_train, y_train (forward), y_train_inv (inverse), forward_table.
    """
    forward_table, inverse_table = generate_substitution_tables(k1, k2)

    X_train = np.eye(256, dtype=np.float32)
    y_train = np.eye(256, dtype=np.float32)[forward_table]

    X_inv = np.eye(256, dtype=np.float32)
    y_inv = np.eye(256, dtype=np.float32)[inverse_table]

    return X_train, y_train, y_inv, forward_table


def train_ann_pair(sample_images: list, k1: int, k2: int,
                   epochs: int = 50, save_dir: str = 'saved_keys') -> Tuple[keras.Model, keras.Model]:
    """
    Train both forward and inverse ANN models.
    
    Args:
        sample_images: List of sample images for training
        k1: Permutation key
        k2: Diffusion key
        epochs: Number of training epochs (default: 50 for optimal convergence)
        save_dir: Directory to save trained models
    
    Returns:
        Tuple of (forward_model, inverse_model)
    """
    print("Preparing training data...")
    X_train, y_train, y_inv, forward_table = prepare_training_data(k1, k2)
    
    # Train forward ANN
    print("\nTraining forward ANN (encryption)...")
    forward_model = create_ann_model()
    forward_model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=64,
        verbose=1,
        validation_split=0.1,
        shuffle=True
    )
    
    # Train inverse ANN (reverse mapping)
    print("\nTraining inverse ANN (decryption)...")
    inverse_model = create_ann_model()
    inverse_model.fit(
        X_train,
        y_inv,
        epochs=epochs,
        batch_size=64,
        verbose=1,
        validation_split=0.1,
        shuffle=True
    )
    
    # Save models
    os.makedirs(save_dir, exist_ok=True)
    forward_model.save(os.path.join(save_dir, 'forward_ann.h5'))
    inverse_model.save(os.path.join(save_dir, 'inverse_ann.h5'))
    np.save(os.path.join(save_dir, 'substitution_table.npy'), forward_table)
    print(f"\nModels saved to {save_dir}/")
    
    return forward_model, inverse_model


def ann_substitute(
    pixels: np.ndarray,
    model: keras.Model,
    substitution_table: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Level 3: Apply ANN-based substitution (encryption).
    If a substitution_table is provided, use it directly for perfect invertibility; otherwise use the ANN.
    """
    pixels_uint8 = pixels.astype(np.uint8).flatten()

    if substitution_table is not None:
        return substitution_table[pixels_uint8]

    one_hot = np.eye(256, dtype=np.float32)[pixels_uint8]
    output = model.predict(one_hot, verbose=0)
    substituted = np.argmax(output, axis=1).astype(np.uint8)
    return substituted


def ann_reverse(
    pixels: np.ndarray,
    inverse_model: keras.Model,
    inverse_table: Optional[np.ndarray] = None,
    substitution_table: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Reverse Level 3: Apply inverse ANN substitution (decryption).
    Prefers inverse_table if provided; otherwise computes from substitution_table; else uses ANN.
    """
    pixels_uint8 = pixels.astype(np.uint8).flatten()

    if inverse_table is None and substitution_table is not None:
        inverse_table = np.argsort(substitution_table).astype(np.uint8)

    if inverse_table is not None:
        return inverse_table[pixels_uint8]

    one_hot = np.eye(256, dtype=np.float32)[pixels_uint8]
    output = inverse_model.predict(one_hot, verbose=0)
    recovered = np.argmax(output, axis=1).astype(np.uint8)
    return recovered


def load_ann_models(save_dir: str = 'saved_keys') -> Tuple[keras.Model, keras.Model]:
    """
    Load pre-trained ANN models.
    
    Args:
        save_dir: Directory containing saved models
    
    Returns:
        Tuple of (forward_model, inverse_model)
    """
    forward_path = os.path.join(save_dir, 'forward_ann.h5')
    inverse_path = os.path.join(save_dir, 'inverse_ann.h5')
    
    forward_model = keras.models.load_model(forward_path)
    inverse_model = keras.models.load_model(inverse_path)
    
    return forward_model, inverse_model


def load_substitution_table(save_dir: str = 'saved_keys') -> Optional[np.ndarray]:
    """Load saved substitution table if available."""
    table_path = os.path.join(save_dir, 'substitution_table.npy')
    if os.path.exists(table_path):
        return np.load(table_path).astype(np.uint8)
    return None
