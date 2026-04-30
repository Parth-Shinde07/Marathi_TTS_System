# Technical Specification: HiFi-GAN Vocoder in Marathi TTS

This document outlines the mathematical foundation and architecture of the HiFi-GAN Vocoder used for converting Mel-spectrograms into high-fidelity Marathi audio waveforms.

## Architecture Overview

The system utilizes a Generative Adversarial Network (GAN) architecture consisting of:

- **Generator (G)**: A sequence of transposed convolutions with multi-receptive field fusion (MRF) that upsamples Mel-spectrograms (using **80 Mel filters**) to time-domain audio.
- **Discriminator (D)**: Comprised of:
    - **Multi-Period Discriminator (MPD)**: Handles periodic patterns (pitch/harmonics).
    - **Multi-Scale Discriminator (MSD)**: Handles consecutive audio patterns (texture/roughness).

## Loss Functions

The total loss during training is defined as:

L_total = L_adv + λ_fm * L_fm + λ_mel * L_mel

### 1. Adversarial Loss (L_adv)
Ensures the generated audio sounds realistic to the discriminator.
- L_adv(G) = E[(D(G(s)) - 1)^2]
- L_adv(D) = E[(D(x) - 1)^2] + E[D(G(s))^2]
*Where x is real audio and s is the source Mel-spectrogram.*

### 2. Feature Matching Loss (L_fm)
Compares the internal hidden representations of the discriminator for real and fake audio. It ensures that "high-level" features like tone and texture match the training data.
- L_fm(G, D) = E [ Σ ||D_layer(x) - D_layer(G(s))||_1 ]

### 3. Mel-Spectrogram Loss (L_mel)
Enforces visual similarity between the source spectrogram and the spectrogram reconstructed from the generated audio. This is critical for pronunciation and content accuracy.
- L_mel(G) = E [ ||Mel(x) - Mel(G(s))||_1 ]

## Configuration (Lambda Values)

The balance between **Naturalness** and **Accuracy** is controlled by the λ weights:

| Weight | Parameter | Current Value | Effect of High Value |
| :--- | :--- | :--- | :--- |
| λ_mel | `mel_loss_alpha` | **45.0** | More accurate pronunciation, avoids mismatch. |
| λ_fm | `feat_loss_alpha` | **1.0** | More natural quality, smoother textures. |

## Training Configuration
| Parameter | Value |
| :--- | :--- |
| **Total Training Steps** | 30,000 |
| **Learning Rate (LR)** | 5 × 10⁻⁵ |

---
*Note: We use a high λ_mel (45.0) for Marathi because the phoneme-to-audio mapping in Devanagari requires strict adherence to the spectral content for intelligibility. The training is conducted for 30,000 steps with a slower learning rate of 5 × 10⁻⁵ to ensure stable convergence and high-fidelity output.*
