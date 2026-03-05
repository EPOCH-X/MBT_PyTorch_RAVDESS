# Multimodal Bottleneck Transformer (MBT) in PyTorch

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

An unofficial PyTorch implementation of the **"Attention Bottlenecks for Multimodal Fusion" (NeurIPS 2021)** architecture. This repository demonstrates audio-visual emotion recognition using the Kaggle RAVDESS dataset, featuring a highly optimized data pipeline for A100 GPUs.

> **본 레포지토리는 멀티모달 병목 트랜스포머(MBT)의 PyTorch 구현체입니다. 비디오와 오디오 데이터를 효율적으로 융합하여 화자의 감정을 인식하며, A100 GPU의 성능을 극대화하기 위한 고속 데이터 전처리 파이프라인이 적용되어 있습니다.**

1) **[Notion Blog Review](https://www.notion.so/MBT-NeurIPS-2021-2ee41bbe12988052b2bed32596030b5a)**
2) **[Tistory Blog Review](https://pak1010pak.tistory.com/131)**
3) **[Test Review](https://pak1010pak.tistory.com/132)**

## 📌 Project Highlights
* **Mid-Fusion with Attention Bottlenecks**: Implemented the core idea of restricting cross-modal interaction through a small number of bottleneck tokens, improving efficiency and performance. (시각/청각 정보가 병목 토큰을 통해서만 교류하도록 강제하여 연산 효율 극대화)
* **High-Speed Data Caching Pipeline**: Solved CPU/IO bottlenecks by pre-processing heavy MP4 decodings and audio spectrograms into cached `.pt` tensors. (에포크마다 발생하는 영상 디코딩 병목을 없애기 위해 사전 텐서 캐싱 파이프라인 구축)
* **High Performance**: Reached **99.18% training accuracy** in just 10 epochs on the RAVDESS dataset. (10 에포크 만에 훈련 정확도 99.18% 달성)

## 📊 Dataset: RAVDESS
* **Source**: [Kaggle RAVDESS Dataset](https://www.kaggle.com/datasets/orvile/ravdess-dataset) (Ryerson Audio-Visual Database of Emotional Speech and Song)
* **Vision**: Extracted 16 RGB frames per video using OpenCV (ResNet18 Feature Extractor).
* **Audio**: Extracted 128x128 Mel-Spectrograms using Librosa (Conv2d Patch Embedding).
* **Classes**: 8 Emotions (Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised).

## 🚀 Training Results
Trained on a single **NVIDIA A100 80GB** GPU. Thanks to the cached dataloader (`num_workers=12`, `pin_memory=True`), each epoch takes only ~37 seconds.

| Epoch | Loss | Accuracy (%) |
| :---: | :---: | :---: |
| 1 | 0.5081 | 84.91 |
| 5 | 0.0512 | 98.41 |
| **10** | **0.0260** | **99.18** |

## 👀 Inference Visualization
The script visualizes the model's prediction process. It displays the input video frame (Vision), the Mel-Spectrogram (Audio), and the model's predicted probability distribution across 8 emotions.

<img width="1132" height="662" alt="image" src="https://github.com/user-attachments/assets/d11b1d6a-6459-4c02-ad4b-d2ecd33618c3" />

* **Green Bar**: Ground Truth (정답)
* **Red Bar**: Incorrect Prediction (오답일 경우 표시됨)

## 🛠️ Getting Started

### 1. Requirements
```bash
pip install torch torchvision torchaudio librosa opencv-python matplotlib tqdm
```

### 2. Prepare the Data
1) Download the RAVDESS .mp4 dataset from Kaggle.
2) Run the preprocessing script to generate cached .pt tensors.

```Python
# The script will create a 'ravdess_preprocessed' folder containing tensor files.
# This prevents CPU bottlenecks during training.
```

### 3. Run the Notebook
Open the provided Jupyter Notebook (MBT_Implementation.ipynb), run all cells to initialize the SimpleMBT model, train it, and visualize the inference results.

## 📝 References
- Paper: [Attention Bottlenecks for Multimodal Fusion (Nagrani et al., 2021)](https://arxiv.org/abs/2107.00135)
