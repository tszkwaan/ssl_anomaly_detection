import pandas as pd
import numpy as np
import torch
import os

# === 設定 ===
WIN_SIZE = 100
STRIDE = 10
DATA_PATH = 'data_preprocessing/PSM'
OUT_PATH = 'data/PSM'

def create_windows(data, window_size, stride):
    if len(data) < window_size:
        return np.empty((0, data.shape[1], window_size))
    n_samples = (len(data) - window_size) // stride + 1
    samples = []
    for i in range(n_samples):
        start = i * stride
        end = start + window_size
        samples.append(data[start:end, :])
    return np.array(samples).transpose(0, 2, 1)

def main():
    print(f"Loading CSV files from {DATA_PATH}...")
    
    # 1. 讀取 CSV (略過第一欄 timestamp)
    # === 關鍵修復：.fillna(0) 強制補洞 ===
    df_train = pd.read_csv(os.path.join(DATA_PATH, 'train.csv')).iloc[:, 1:].fillna(0)
    df_val   = pd.read_csv(os.path.join(DATA_PATH, 'val.csv')).iloc[:, 1:].fillna(0)
    df_test  = pd.read_csv(os.path.join(DATA_PATH, 'test.csv')).iloc[:, 1:].fillna(0)
    
    # 讀取 Labels
    # 注意：如果沒有 label 檔，請自行生成全 0 的 dummy label
    if os.path.exists(os.path.join(DATA_PATH, 'test_label.csv')):
        df_test_labels = pd.read_csv(os.path.join(DATA_PATH, 'test_label.csv')).iloc[:, 1:].fillna(0)
        test_labels_vals = df_test_labels.values.flatten().astype(int)
    else:
        print("Warning: test_label.csv not found, using dummy labels.")
        test_labels_vals = np.zeros(len(df_test))

    # 2. 轉 Numpy 並強力清洗 (處理 Infinity)
    # === 關鍵修復：np.nan_to_num 處理無限大 ===
    train_vals = np.nan_to_num(df_train.values.astype(np.float32), posinf=0, neginf=0)
    val_vals   = np.nan_to_num(df_val.values.astype(np.float32), posinf=0, neginf=0)
    test_vals  = np.nan_to_num(df_test.values.astype(np.float32), posinf=0, neginf=0)

    print(f"Train NaN check: {np.isnan(train_vals).any()}") # 這裡應該要顯示 False

    # 3. 正規化 (Normalize)
    mean = np.mean(train_vals, axis=0)
    std = np.std(train_vals, axis=0)
    std[std < 1e-5] = 1.0 # 避免除以 0

    train_norm = (train_vals - mean) / std
    val_norm   = (val_vals - mean) / std
    test_norm  = (test_vals - mean) / std
    
    # 二次清洗
    train_norm = np.nan_to_num(train_norm)
    val_norm   = np.nan_to_num(val_norm)
    test_norm  = np.nan_to_num(test_norm)

    # 4. 切分視窗
    print("Slicing windows...")
    X_train = create_windows(train_norm, WIN_SIZE, STRIDE)
    y_train = np.zeros(len(X_train))
    
    X_val = create_windows(val_norm, WIN_SIZE, STRIDE)
    y_val = np.zeros(len(X_val))
    
    X_test = create_windows(test_norm, WIN_SIZE, WIN_SIZE)
    
    y_test = []
    n_test_windows = len(X_test)
    for i in range(n_test_windows):
        s = i * WIN_SIZE
        e = s + WIN_SIZE
        if e > len(test_labels_vals): break
        label = 1 if np.sum(test_labels_vals[s:e]) > 0 else 0
        y_test.append(label)
    y_test = np.array(y_test)

    # 5. 存檔
    if not os.path.exists(OUT_PATH): os.makedirs(OUT_PATH)
    print(f"Saving cleaned data to {OUT_PATH}...")
    torch.save({'samples': torch.from_numpy(X_train).float(), 'labels': torch.from_numpy(y_train).long()}, os.path.join(OUT_PATH, 'train.pt'))
    torch.save({'samples': torch.from_numpy(X_val).float(), 'labels': torch.from_numpy(y_val).long()}, os.path.join(OUT_PATH, 'val.pt'))
    torch.save({'samples': torch.from_numpy(X_test).float(), 'labels': torch.from_numpy(y_test).long()}, os.path.join(OUT_PATH, 'test.pt'))
    print("✅ 數據修復完成！")

if __name__ == '__main__':
    main()