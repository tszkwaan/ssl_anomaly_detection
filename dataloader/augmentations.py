import numpy as np
import torch

def DataTransform(sample, config):
    # 確保輸入轉為 numpy
    if isinstance(sample, torch.Tensor):
        sample = sample.numpy()
        
    weak_aug = scaling(sample, config.augmentation.jitter_scale_ratio)
    strong_aug = jitter(permutation(sample, max_segments=config.augmentation.max_seg), config.augmentation.jitter_ratio)
    
    return torch.from_numpy(weak_aug).float(), torch.from_numpy(strong_aug).float()

def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    # 限制噪聲強度，避免破壞標準化後的數據
    noise = np.random.normal(loc=0., scale=sigma, size=x.shape)
    # 將噪聲限制在合理範圍內，避免極端值
    noise = np.clip(noise, -3*sigma, 3*sigma)
    return x + noise

def scaling(x, sigma=1.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    # x shape: (Samples, Channels, Time) or (Channels, Time)
    # 我們生成一個隨機 scaling factor，維度要能跟 x 廣播 (Broadcast)
    if len(x.shape) == 3: # (N, C, L)
        factor = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], x.shape[1], 1))
    else: # (C, L)
        factor = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], 1))
    
    # 確保 factor 為正數，並限制在合理範圍內 [0.5, 2.0]，避免數值問題
    factor = np.abs(factor)
    factor = np.clip(factor, 0.5, 2.0)
    return x * factor

def permutation(x, max_segments=5, seg_mode="equal"):
    # x shape: (N, C, L) or (C, L)
    orig_shape = x.shape
    
    # 如果輸入是單個樣本 (C, L)，擴展為 (1, C, L) 以便統一處理
    if len(orig_shape) == 2:
        x = x[np.newaxis, :, :]
    
    N, C, L = x.shape
    ret = np.zeros_like(x)
    
    # 為每個樣本生成隨機切分段數
    num_segs = np.random.randint(1, max_segments + 1, size=N)
    
    for i in range(N):
        if num_segs[i] > 1:
            # 切分
            if seg_mode == "random":
                split_points = np.random.choice(L - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(x[i], split_points, axis=1)
            else:
                splits = np.array_split(x[i], num_segs[i], axis=1)
            
            # === 關鍵修正：不直接 permute splits，而是 permute 索引 ===
            # 這解決了 "inhomogeneous shape" 的錯誤
            perm_indices = np.random.permutation(len(splits))
            shuffled_splits = [splits[j] for j in perm_indices]
            
            # 重組
            warp = np.concatenate(shuffled_splits, axis=1)
            ret[i] = warp
        else:
            ret[i] = x[i]
            
    # 如果原本是 2D，還原回去
    if len(orig_shape) == 2:
        return ret[0]
    return ret