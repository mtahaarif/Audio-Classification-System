# Audio Classification: Speech, Music, and Noise

This repository contains a robust audio classification system designed to categorize signals into three classes: **Speech**, **Music**, and **Noise**. The project leverages Digital Signal Processing (DSP) for signal conditioning and a Deep Learning (DL) classifier for high-accuracy inference.

## 🛠️ Signal Processing Pipeline

The pre-processing stage ensures the removal of artifacts and emphasizes class-specific frequency components:

### 1. Signal Conditioning (MATLAB)
* **Speech Enhancement**: Applied a **Band-Pass Filter (500–6000 Hz)** to target the human vocal range.
* **Music Conditioning**: Applied a **Low-Pass Filter (12,000 Hz)** to eliminate high-frequency noise while preserving melodic content.
* **Normalization**: Signals were converted to mono and normalized to ensure uniform amplitude distribution across the dataset.

### 2. Feature Extraction (Python/Librosa)
Five critical temporal and spectral features were extracted to create the input feature vector:
* **MFCCs (Mel-frequency Cepstral Coefficients)**: 13 coefficients to capture the spectral envelope.
* **Zero Crossing Rate (ZCR)**: To distinguish between voiced speech (low ZCR) and percussive noise (high ZCR).
* **Root Mean Square (RMS) & Energy**: To analyze the signal's power and intensity.
* **Pitch Detection**: Used to identify fundamental frequencies characteristic of melodic signals.

## 🧠 Deep Learning Architecture

The classification core is a **Multilayer Perceptron (MLP)** built using **TensorFlow/Keras**:

* **Input Layer**: 17-dimensional feature vector (MFCCs + ZCR + RMS + Energy + Pitch).
* **Hidden Layers**: Dense layers with **ReLU** activation and **Dropout** (0.2) to mitigate overfitting.
* **Output Layer**: Softmax activation for 3-class categorization.
* **Optimizer**: Adam with Categorical Crossentropy loss.

### Performance Metrics
* **Training Accuracy**: 98.85%
* **Testing Accuracy**: 94.35%

## 🚀 Deployment & UI

The system is deployed via a **Flask web application** that provides:
1.  **Audio Upload**: Direct processing of `.wav` files.
2.  **Spectrogram Analysis**: Real-time generation of the signal's spectrogram using `matplotlib` and `librosa`.
3.  **Visual Comparison**: The UI displays the uploaded signal's spectrogram alongside reference spectrograms for Music, Speech, and Noise for visual verification.

## 📁 Dataset
The model was trained and validated using the **MUSAN (Music, Speech, and Noise) dataset**, ensuring a diverse range of acoustic environments and recording qualities.
