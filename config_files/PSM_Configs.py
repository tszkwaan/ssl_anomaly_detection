class Config(object):
    def __init__(self):
        # ==========================
        # 1. 模型結構參數 (Model Configs)
        # ==========================
        self.input_channels = 25   # PSM 數據集的特徵數 (必須是 26)
        self.kernel_size = 8       # 卷積核大小，配合 Window=100 設小一點
        self.stride = 1            # 步長
        self.final_out_channels = 128 # 編碼器輸出的特徵深度

        self.num_classes = 2       # 0:正常, 1:異常
        self.dropout = 0.35        
        
        # features_len 是 CNN 輸出後的特徵時間長度
        # 輸入 WIN_SIZE -> 經過 3 層 CNN + 3 層 MaxPool -> 輸出長度
        # WIN_SIZE=100 -> features_len≈15, WIN_SIZE=256 -> features_len≈34
        # 注意：evaluate_anomaly.py 會自動計算正確的值，這裡只是預設值
        self.features_len = 15    

        # ==========================
        # 2. 訓練參數 (Training Configs)
        # ==========================
        self.num_epoch = 40        # 訓練輪數，40 輪對 PSM 應該足夠收斂

        # ==========================
        # 3. 優化器參數 (Optimizer Parameters)
        # ==========================
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4             # 學習率

        # ==========================
        # 4. 數據加載參數 (Data Parameters)
        # ==========================
        self.drop_last = True      # 如果最後一批數據不足 batch_size 則丟棄
        self.batch_size = 32       # 建議：如果您的顯存不大，先設 32；如果不夠跑再改 16

        # ==========================
        # 5. 子模組配置 (Sub-modules)
        # (這些絕對不能刪！)
        # ==========================
        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()


# ==========================================
# 數據增強配置 (Augmentations)
# ==========================================
class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 1.1 # 縮放強度的參數
        self.jitter_ratio = 0.8       # 加噪聲的參數
        self.max_seg = 8              # Permutation 切割的最大段數


# ==========================================
# 上下文對比配置 (Contextual Contrasting)
# ==========================================
class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.1          # InfoNCE Loss 的溫度參數
        self.use_cosine_similarity = True # 使用餘弦相似度


# ==========================================
# 時序對比配置 (Temporal Contrasting)
# ==========================================
class TC(object):
    def __init__(self):
        self.hidden_dim = 100   # TC 模組內部的隱藏層維度
        self.timesteps = 6      # 預測未來幾個時間步 (不能大於 features_len)
                                # 建議設小一點 (例如 6-10)，因為 features_len 只有 24