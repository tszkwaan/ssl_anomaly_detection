# TS-TCC for Anomaly Detection on PSM Dataset

This repository contains an implementation of **Time-Series Representation Learning via Temporal and Contextual Contrasting (TS-TCC)** adapted for anomaly detection on the **PSM (Pooled Server Metrics)** dataset.

## Overview

This project applies the TS-TCC self-supervised learning framework to learn robust time-series representations from unlabeled server monitoring data, then uses these representations for anomaly detection. The model achieves **F1-Score: 0.8161, Precision: 0.8351, Recall: 0.7979, AUROC: 0.6228** on the PSM dataset.

## Dataset

**PSM (Pooled Server Metrics)** is a server monitoring dataset containing:
- **25 channels** of server metrics (CPU, memory, network, etc.)
- **Window size**: 100 timesteps
- **Task**: Binary anomaly detection (normal vs. anomaly)

The dataset is preprocessed into sliding windows with stride=10 for training and stride=100 for testing.

## Model: TS-TCC

**Time-Series Temporal and Contextual Contrasting (TS-TCC)** is a self-supervised learning framework that:

1. **Temporal Contrasting**: Learns robust temporal representations by predicting future timesteps across different augmented views
2. **Contextual Contrasting**: Learns discriminative representations by maximizing similarity among contexts of the same sample while minimizing similarity among different samples

The model architecture consists of:
- **Encoder**: 3-layer CNN with MaxPooling
- **Temporal Contrasting Module**: LSTM-based predictor
- **Contextual Contrasting Module**: Contrastive learning with InfoNCE loss

## Key Modifications

This implementation includes several improvements over the original TS-TCC:

### 1. **Feature Extraction for Anomaly Detection**
- **Mean Pooling**: Applied on temporal dimension after CNN encoding to extract fixed-size feature vectors
- **Batch Processing**: Implemented batch-wise feature extraction to handle large datasets efficiently
- **Correct Feature Usage**: Uses convolutional features (second return value) instead of logits for anomaly detection

### 2. **Data Preprocessing Enhancements**
- **NaN/Inf Handling**: Comprehensive cleaning of missing and infinite values
- **Robust Normalization**: Z-score normalization with zero-division protection
- **Data Validation**: Checks for data format consistency and missing files

### 3. **Training Stability**
- **Gradient Clipping**: Prevents gradient explosion (max_norm=1.0)
- **NaN/Inf Detection**: Skips batches with invalid loss values
- **Progress Tracking**: Real-time training progress with tqdm progress bars

### 4. **Anomaly Detection Pipeline**
- **Cosine Similarity**: Uses cosine distance between test features and training data center
- **Point Adjustment**: Implements point adjustment strategy for evaluation
- **Threshold Search**: Brute-force search for optimal F1-Score threshold

### 5. **Configuration**
- **Temperature**: Set to 0.1 for InfoNCE loss (tighter contrastive learning)
- **Training Epochs**: 40 epochs for sufficient convergence
- **Batch Size**: 32 (adjustable based on GPU memory)

## Setup Instructions

### Step 1: Data Preprocessing

The processed PSM dataset CSV files are located in `data_preprocessing/PSM/`:
- `train.csv`
- `val.csv`
- `test.csv`
- `test_label.csv` (optional, will use dummy labels if not found)

You need to run the preprocessing script to generate `.pt` files:

```bash
python data_preprocessing/PSM/generate_PSM.py
```

This will:
- Load CSV files (skip first column as timestamp)
- Handle NaN/Inf values
- Normalize data using Z-score
- Create sliding windows (WIN_SIZE=100, STRIDE=10 for train/val, STRIDE=100 for test)
- Save preprocessed data as `.pt` files in `data/PSM/`

**Output files:**
- `data/PSM/train.pt`
- `data/PSM/val.pt`
- `data/PSM/test.pt`

### Step 2: Upload to Google Drive

1. Upload the entire project folder to your Google Drive
2. Note the path (e.g., `/content/drive/MyDrive/bosch/TS-TCC`)

### Step 3: Run on Google Colab

1. **Open the notebook**: `ssl_trial.ipynb` in Google Colab

2. **Update the project path** in Cell 0:
   ```python
   project_path = '/content/drive/MyDrive/YOUR_PATH/TS-TCC'
   ```

3. **Run Cell 0**: This will:
   - Mount Google Drive
   - Change to project directory
   - Verify required files exist
   - Install dependencies

4. **Run Cell 1**: This will:
   - Check GPU availability
   - Start training with self-supervised mode
   - Training will take ~40 epochs (adjustable in `PSM_Configs.py`)

5. **Run Cell 3**: After training completes, this will:
   - Automatically find the latest checkpoint
   - Load preprocessed data
   - Extract features using the trained model
   - Calculate anomaly scores
   - Evaluate and visualize results

## Project Structure

```
TS-TCC/
├── config_files/
│   └── PSM_Configs.py          # Configuration for PSM dataset
├── data/
│   └── PSM/
│       ├── train.pt            # Preprocessed training data
│       ├── val.pt              # Preprocessed validation data
│       └── test.pt             # Preprocessed test data
├── data_preprocessing/
│   └── PSM/
│       ├── generate_PSM.py    # Data preprocessing script
│       ├── train.csv           # Raw training data
│       ├── val.csv             # Raw validation data
│       └── test.csv            # Raw test data
├── dataloader/
│   ├── dataloader.py           # Data loading utilities
│   └── augmentations.py       # Data augmentation functions
├── models/
│   ├── model.py                # Base encoder model
│   ├── TC.py                   # Temporal Contrasting module
│   ├── attention.py            # Attention mechanisms
│   └── loss.py                 # Loss functions
├── trainer/
│   └── trainer.py              # Training and evaluation loops
├── main.py                     # Main training script
├── ssl_trial.ipynb             # Colab notebook for training and evaluation
└── README.md                   # This file
```

## Configuration

Key parameters in `config_files/PSM_Configs.py`:

- `input_channels`: 25 (PSM dataset features)
- `features_len`: 15 (CNN output temporal length for WIN_SIZE=100)
- `num_epoch`: 40 (training epochs)
- `batch_size`: 32 (adjust based on GPU memory)
- `temperature`: 0.1 (InfoNCE loss temperature)
- `timesteps`: 6 (temporal prediction steps)

## Results

On PSM dataset with the current configuration:
- **F1-Score**: 0.8161
- **Precision**: 0.8351
- **Recall**: 0.7979
- **AUROC**: 0.6228
- **Best Threshold**: 0.432079

## Acknowledgments

- Original TS-TCC implementation: [emadeldeen24/TS-TCC](https://github.com/emadeldeen24/TS-TCC)
- PSM dataset for server monitoring anomaly detection (https://github.com/eBay/RANSynCoders/tree/main)
