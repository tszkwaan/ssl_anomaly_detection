import numpy as np
import torch

def DataTransform(sample, config):
    # Ensure input is converted to numpy
    if isinstance(sample, torch.Tensor):
        sample = sample.numpy()
        
    weak_aug = scaling(sample, config.augmentation.jitter_scale_ratio)
    strong_aug = jitter(permutation(sample, max_segments=config.augmentation.max_seg), config.augmentation.jitter_ratio)
    
    return torch.from_numpy(weak_aug).float(), torch.from_numpy(strong_aug).float()

def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    # Limit noise intensity to avoid corrupting normalized data
    noise = np.random.normal(loc=0., scale=sigma, size=x.shape)
    # Clip noise to reasonable range to avoid extreme values
    noise = np.clip(noise, -3*sigma, 3*sigma)
    return x + noise

def scaling(x, sigma=1.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    # x shape: (Samples, Channels, Time) or (Channels, Time)
    # Generate a random scaling factor with dimensions that can broadcast with x
    if len(x.shape) == 3: # (N, C, L)
        factor = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], x.shape[1], 1))
    else: # (C, L)
        factor = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], 1))
    
    # Ensure factor is positive and limit to reasonable range [0.5, 2.0] to avoid numerical issues
    factor = np.abs(factor)
    factor = np.clip(factor, 0.5, 2.0)
    return x * factor

def permutation(x, max_segments=5, seg_mode="equal"):
    # x shape: (N, C, L) or (C, L)
    orig_shape = x.shape
    
    # If input is a single sample (C, L), expand to (1, C, L) for unified processing
    if len(orig_shape) == 2:
        x = x[np.newaxis, :, :]
    
    N, C, L = x.shape
    ret = np.zeros_like(x)
    
    # Generate random number of segments for each sample
    num_segs = np.random.randint(1, max_segments + 1, size=N)
    
    for i in range(N):
        if num_segs[i] > 1:
            # Split
            if seg_mode == "random":
                split_points = np.random.choice(L - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(x[i], split_points, axis=1)
            else:
                splits = np.array_split(x[i], num_segs[i], axis=1)
            
            # === Key fix: Permute indices instead of directly permuting splits ===
            # This fixes the "inhomogeneous shape" error
            perm_indices = np.random.permutation(len(splits))
            shuffled_splits = [splits[j] for j in perm_indices]
            
            # Reassemble
            warp = np.concatenate(shuffled_splits, axis=1)
            ret[i] = warp
        else:
            ret[i] = x[i]
            
    # If originally 2D, restore it
    if len(orig_shape) == 2:
        return ret[0]
    return ret