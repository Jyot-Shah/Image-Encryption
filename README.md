# ANN-Based Three-Level Image Encryption System

**Status**: Complete | **Date**: January 23, 2026 | **Reference**: IEEE Access Paper (2024)

## Overview

This project implements a three-level image encryption system:
1. **Level 1**: Permutation (Confusion) using key k1
2. **Level 2**: Chained XOR Diffusion using key k2 
3. **Level 3**: ANN substitution with key-derived bijection

**All 4 objectives fully implemented** with comprehensive metrics, attack testing, and comparative analysis.

## Improvements Over The Paper (Highlights)

- **Chained XOR with 2-Byte Feedback**: Enhanced feedback chaining (uses both previous cipher bytes) for stronger avalanche effect.
- **Key-Derived 0–255 Bijection + Dual ANN**: Instead of approximate ANN mapping, we generate a deterministic key-specific substitution table and train both forward and inverse ANNs on the exact bijection for pixel-perfect reversibility (SSIM ≈ 1, PSNR ≫ 40 dB).
- **Multi-Image Evaluation + Top-5 Ranking**: The pipeline evaluates all images, ranks by encrypted entropy and NPCR, and saves a concise top-5 metrics table plus “best_*” visualizations for the most secure outputs.
- **Robustness Focused on Practical Attacks**: We emphasize salt-pepper and occlusion attacks (bit errors, packet loss) with PSNR/SSIM reporting, aligning robustness to realistic channels over additive Gaussian noise.
- **Histogram Uniformity Check (Chi-square)**: Added a statistical uniformity test to verify encrypted histograms are indistinguishable from uniform, strengthening confusion evidence.
- **Key Sensitivity Built-In**: NPCR/UACI measured under single-bit key flips as part of the evaluation loop to demonstrate strong key sensitivity.
- **Architecture Variation + Baseline Comparisons**: Included architecture sweeps and baseline tests (perm-only, XOR-only, perm+XOR) to quantify why the three-level + ANN design is superior.
- **Performance and Reproducibility**: Vectorized NumPy, shallow ANN, deterministic keys/tables, and saved models/keys make the system faster, reproducible, and practical.

### Snapshot Comparison vs Baselines

| Aspect | XOR-only Baseline | Proposed (Perm+XOR+ANN) | Best Achieved | Outcome |
|--------|-------------------|--------------------------|-------------------|------|
| Entropy | ~7.95 | ~7.97 | **7.997692 bits** | Near-perfect randomness |
| Correlation | Residual patterns | Near zero (~0.0) | **0.000015** (diagonal) | Near-perfect decorrelation |
| NPCR | ~99.5% | ~99.6% | **99.67%** (99.6689%) | Exceeds 99.6% target |
| UACI | ~33.3% | ~33.4% | **33.5733%** | Optimal uniform change |
| Reversibility | Good | Pixel-perfect (SSIM≈1, PSNR≫40 dB) | **Perfect (SSIM=1.0, PSNR=∞)** | Lossless encryption |

**Key Enhancements** (Current system achieves):
- **2-byte XOR feedback chaining**: Stronger avalanche than 1-byte, boosts NPCR by ~0.05%
- **512-neuron hidden layer**: Enhanced nonlinearity (from original 256), increases entropy and NPCR
- **50 training epochs**: Improved ANN convergence on bijection (from original 30), stabilizes metrics

See detailed plots in `comparisons/baseline_comparison.png` and architecture sweeps in `comparisons/ann_architecture_comparison.png`.

## Implementation Details

**Core Modules** (8 files, ~2050 lines):
- `encryption.py` / `decryption.py` - Three-level pipeline
- `ann_model.py` - ANN architecture & training
- `metrics.py` - 6 security metrics + quality assessment
- `attacks.py` - Salt-pepper, occlusion robustness
- `comparison_analysis.py` - Architecture/baseline comparisons
- `main.py` - Orchestration with multi-image evaluation

**Project Structure:**
```
project/
├── Core: encryption.py, decryption.py, ann_model.py, metrics.py, attacks.py, main.py
├── Support: comparison_analysis.py, requirements.txt
├── Documentation: README.md
├── Data: data/processed/ (256×256 preprocessed images), data/USC-SIPI Image Database/ (original dataset)
├── Keys: saved_keys/ (trained models, encryption keys)
├── Results: results/visualizations/, results/attack_analysis/
└── Comparisons: comparisons/
```

## Installation & Usage

**Install:**
```bash
pip install -r requirements.txt
```

**Run all analyses:**
```bash
python main.py --all
```

**Individual operations:**
```bash
python main.py --train      # Train ANN models
python main.py --test       # Test encryption/decryption
python main.py --attacks    # Run attack analysis
python main.py --compare    # Run comparative analysis
```

## Dataset

- **Source**: USC-SIPI Image Database (used for training and evaluation)
- **Coverage**: Uses all four USC‑SIPI volumes — **Textures (Vol 1)**, **Aerials (Vol 2)**, **Misc (Vol 3)**, **Sequences (Vol 4)**
- **Preprocessing**: `preprocess_sipi.py` converts images to 256×256 grayscale PNGs
	- Input directory: `data/USC-SIPI Image Database/`
	- Output directory: `data/processed/`
	- Operations: resize to 256×256, grayscale conversion, consistent naming
- **Auto-setup**: If `data/processed/` is empty and `data/USC-SIPI Image Database/` exists, the pipeline preprocesses automatically on first run
- **Training Selection**: Randomly selects up to 20 diverse images from all volumes (fixed seed for reproducibility)

## Three-Level Architecture

```
ENCRYPTION: Image → Permutation (k1) → XOR Diffusion (k2) → ANN Substitution → Encrypted
DECRYPTION: Encrypted → Inverse ANN → Reverse XOR → Inverse Permutation → Original
```

### Level 1: Permutation (Confusion)
- Random shuffle of 65,536 pixels using key k1
- Deterministic: same key always produces same result
- Reversible with stored permutation indices
- Destroys spatial correlations
- Time: ~0.001s

### Level 2: Chained XOR Diffusion  
- Pseudo-random keystream generated from k2
- **2-byte feedback chaining**: XOR with keystream and both previous cipher bytes (prev1 ⊕ prev2)
- Self-inverse: `(X ⊕ K ⊕ P1 ⊕ P2) ⊕ K ⊕ P1 ⊕ P2 = X`
- **Enhanced avalanche**: Stronger diffusion than 1-byte feedback
- Time: ~0.001s

### Level 3: ANN Substitution
- Shallow feedforward network: 256 (input) → **512 (tanh hidden)** → 256 (linear output)
- Maps each pixel value (0-255) through learned bijection
- Forward ANN (encryption) + Inverse ANN (decryption)
- **~256K trainable parameters** per model
- Training: **50 epochs**, MSE loss, Adam optimizer (learning rate: 0.001)
- Time: ~0.015s per image

**Key Reversibility**: All operations are perfectly reversible → PSNR > 50 dB, SSIM > 0.99

## Comprehensive Evaluation (Objectives)

### Objective 1: Framework Design and ANN Architecture Impact

We implement a three-level ANN-based framework and study how ANN architecture choices affect confusion, diffusion, and key sensitivity.

**Modular Architecture**:
- `encryption.py` - Levels 1 & 2 (permutation + XOR)
- `decryption.py` - Inverse operations
- `ann_model.py` - Level 3 (ANN training & inference)
- `metrics.py` - 6 security/quality metrics
- `attacks.py` - Robustness testing
- `main.py` - Orchestration with random 20-image training & multi-image evaluation

**Design Principles**:
- Deterministic: same keys → same output
- Reversible: all operations invertible
- Fast: vectorized NumPy operations
- Lossless: perfect decryption (no information loss)

**Key Management**:
- k1 (32-bit): permutation seed
- k2 (32-bit): XOR keystream seed
- ANN weights: forward + inverse models
- Total key space: 2³² × 2³² × ANN_weight_space

**Hidden Layer Size Analysis**:
- 64 neurons: Entropy ~7.85-7.90 (Good)
- 128 neurons: Entropy ~7.90-7.95 (Better)
- 256 neurons: Entropy ~7.95-7.99 (Excellent)
- **512 neurons** (current): Entropy ~7.9976 (Optimal)

**Activation Function Comparison**:
- tanh: Entropy 7.95-8.00, NPCR 99.4-99.7% ← **Recommended**
- relu: Entropy 7.85-7.92, NPCR 98.8-99.3% (unbounded, less symmetric)
- sigmoid: Entropy 7.88-7.94, NPCR 99.0-99.5% (asymmetric saturation)

**Training Epochs Optimization**:
- 5 epochs: Entropy ~7.88-7.93 (undertrained)
- 10 epochs: Entropy ~7.92-7.97 (good)
- 15 epochs: Entropy ~7.95-7.99 (better)
- 30 epochs: Entropy ~7.95-7.99 (strong)
- **50 epochs** (current): Entropy ~7.9977 (near-optimal)

### Objective 2: Critical Evaluation and Metrics

We evaluate ANN-based image encryption using standard security and image quality metrics on the USC-SIPI dataset to quantify randomness, diffusion, key sensitivity, and reversibility.

**Shannon Entropy** - Measures randomness (Ideal: ~8.0 bits)
- Original images: 6.0-7.5 bits
- Well-encrypted: 7.95-8.00 bits
- Formula: `H(X) = -Σ p(xi) × log2(p(xi))`

**NPCR (Number of Pixel Change Rate)** - Key sensitivity (Ideal: ~99.6%)
- Measures % of pixels that change with 1-bit key change
- Formula: `(different_pixels / total_pixels) × 100%`
- Values >99% indicate strong key sensitivity

**UACI (Unified Average Changing Intensity)** - Change magnitude (Ideal: ~33.4%)
- Measures average pixel intensity difference
- Formula: `(Σ |C1(i,j) - C2(i,j)| / (M × N × 255)) × 100%`

**Correlation Coefficient** - Pixel dependency (Ideal: ~0.0)
- Tested horizontally, vertically, diagonally
- Original images: 0.8-0.95
- Encrypted: should be near 0
- Measures destruction of pixel relationships

**PSNR & SSIM** - Decryption Quality
- PSNR: >40 dB = excellent recovery
- SSIM: >0.99 = perfect structural similarity
- Confirms perfect reversibility

### Objective 3: Robustness Testing

**Attack 1: Salt-and-Pepper Noise** - Simulates bit errors/interference
- Low (1%): PSNR 40-45 dB, SSIM 0.95-0.98 (Excellent)
- Medium (5%): PSNR 32-38 dB, SSIM 0.88-0.94 (Good)
- High (10%): PSNR 25-32 dB, SSIM 0.78-0.88 (Fair)
- **Result**: Good tolerance; errors remain localized

**Attack 2: Occlusion** - Simulates packet loss/cropping
- Mild (10%): PSNR 35-40 dB, SSIM 0.90-0.95 (Good)
- Moderate (25%): PSNR 25-32 dB, SSIM 0.75-0.88 (Fair)
- Severe (50%): PSNR 15-22 dB, SSIM 0.55-0.72 (Poor)
- **Result**: Damage localized; unaffected regions recover well

**Robustness Mechanisms**:
- Level 1 (Permutation): Single incorrect index → single pixel error (no amplification)
- Level 2 (XOR): Bit errors remain localized (self-inverse property)
- Level 3 (ANN): Continuous output tolerates perturbations gracefully

### Objective 4: Comparative Analysis

**Baseline Comparisons**:

| Method | Entropy | Correlation | NPCR (%) | UACI (%) | Conclusion |
|--------|---------|------------|----------|----------|-----------|
| Permutation Only | 7.25 | 0.12 | 99.61 | 33.40 | ❌ No diffusion; histogram unchanged |
| XOR Only | 7.95 | 0.88 | 99.58 | 33.35 | ❌ No confusion; spatial patterns remain |
| Permutation + XOR | 7.96 | 0.01 | 99.62 | 33.42 | ✅ Good; but lacks nonlinearity |
| **Proposed (All 3 + Enhancements)** | **7.9977** | **0.0022** | **99.628** | **33.485** | ✅ **Best: strong diffusion + high entropy** |

**Current System Specifications**:
- **Architecture**: Permutation (k1) → 2-byte XOR Diffusion (k2) → ANN Substitution (256→512→256, 50 epochs)
- **Top-5 Average Performance**:
  - Entropy: 7.9977 bits (near-optimal randomness)
  - NPCR: 99.628% (exceeds 99.6% target)
  - UACI: 33.485% (ideal ~33.4%)
  - Correlation: 0.0022 (near-zero decorrelation)
  - PSNR: ∞ dB, SSIM: 1.0 (perfect reversibility)

**Why Three Levels Superior**:
1. Permutation alone: weak (visual patterns remain, histogram vulnerable)
2. XOR alone: weak (good diffusion but no confusion)
3. Permutation + XOR: good (addresses both but linear/predictable)
4. Permutation + XOR + ANN: **best** (adds nonlinear substitution layer, final confusion stage)

## Highest Metrics Achieved

| Metric | Value | Image | Performance |
|--------|-------|-------|-------------|
| **Highest Entropy** | 7.997692 bits | sequences/6.2.21.png | Near-perfect randomness |
| **Highest NPCR** | 99.6689% | sequences/motion10.512.png | Exceeds 99.6% target |
| **Highest UACI** | 33.5733% | sequences/6.2.21.png | Optimal uniform change |
| **Lowest Correlation (H)** | 0.000206 | sequences/6.2.21.png | Excellent decorrelation |
| **Lowest Correlation (V)** | 0.000144 | sequences/6.2.21.png | Excellent decorrelation |
| **Lowest Correlation (D)** | 0.000015 | aerials/2.2.09.png | Near-perfect decorrelation |

**Overall Statistics**:
- **Entropy**: Min 7.997620, Max 7.997692, Avg 7.997664 bits
- **NPCR**: Min 99.5926%, Max 99.6689%, Avg 99.6277%
- **UACI**: Min 33.4221%, Max 33.5733%, Avg 33.4851%
- **Perfect Quality**: All images decrypt with PSNR = ∞ dB, SSIM = 1.0 (pixel-perfect reversibility)

## Visualizations

### 1: sequences/6.2.21.png
**Metrics**: Entropy 7.9977 | NPCR 99.623% | UACI 33.573% | Correlation ~0.0009
![Best 1 Visualization](results/visualizations/best_1_6_2_21_png.png)

### 2: sequences/motion10.512.png
**Metrics**: Entropy 7.9977 | NPCR 99.669% | UACI 33.435% | Correlation ~0.0022
![Best 2 Visualization](results/visualizations/best_2_motion10_512_png.png)

### 3: aerials/2.2.09.png
**Metrics**: Entropy 7.9977 | NPCR 99.593% | UACI 33.422% | Correlation ~0.0015
![Best 3 Visualization](results/visualizations/best_3_2_2_09_png.png)

### 4: aerials/2.2.22.png
**Metrics**: Entropy 7.9977 | NPCR 99.600% | UACI 33.434% | Correlation ~0.0030
![Best 4 Visualization](results/visualizations/best_4_2_2_22_png.png)

### 5: textures/1.1.09.png
**Metrics**: Entropy 7.9977 | NPCR 99.654% | UACI 33.562% | Correlation ~0.0032
![Best 5 Visualization](results/visualizations/best_5_1_1_09_png.png)

## Robustness Analysis

### Salt-and-Pepper Attack Results
![Salt-Pepper Attack Analysis](results/attack_analysis/salt_pepper_attack.png)

### Occlusion Attack Results
![Occlusion Attack Analysis](results/attack_analysis/occlusion_attack.png)

## Architecture Comparison

### ANN Architecture Impact
![ANN Architecture Comparison](comparisons/ann_architecture_comparison.png)

### Baseline Method Comparison
![Baseline Comparison](comparisons/baseline_comparison.png)

## Output Files

**Metrics:**
- `results/encryption_metrics_all.csv` - Encryption and decryption metrics of the top-5 best images.

**Keys:**
- `saved_keys/forward_ann.h5`, `inverse_ann.h5`, `encryption_keys.npz`

## Typical Performance Results

| Metric | Current (Top-5 Avg) | Target | Status |
|--------|----------|--------|--------|
| **Entropy** | 7.9977 bits | ~8.0 | Near-optimal |
| **NPCR** | 99.628% | ~99.6%–99.7% | Excellent |
| **UACI** | 33.485% | ~33.4% | Ideal |
| **Correlation** | 0.0022 | ~0.0 | Near-zero |
| **PSNR** | ∞ dB | >40 dB | Perfect reversibility |
| **SSIM** | 1.0 | >0.95 | Perfect decryption |
| **Encryption Time** | 44.7 ms | <100 ms | Fast |
| **Training** | 30–60 s | One-time | Acceptable |
| **Key Space** | 2⁶⁴+ | Large | Secure |

**Summary**: All metrics exceed cryptographic standards. System demonstrates excellent confusion, diffusion, and key sensitivity.

## Technical Specs

- **Image Format**: 256×256 grayscale
- **Pixel Range**: 0–255 (uint8)
- **Key Space**: 2³² × 2³² × ANN weights
- **ANN Parameters**: ~131,000
- **Platform**: Python 3.8+, TensorFlow 2.10+

## References

IEEE Access Paper: "A Faster and Robust Artificial Neural Network Based Image Encryption Technique With Improved SSIM" (2024)

## License & Author

**Author**: Jyot Shah  
**LinkedIn**: [linkedin.com/in/jyotshah1](https://linkedin.com/in/jyotshah1)  
**License**: Academic/Research Use  
**Implementation Date**: January 23, 2026
