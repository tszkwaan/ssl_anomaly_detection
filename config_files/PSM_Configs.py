class Config(object):
    def __init__(self):
        # ==========================
        # 1. Model Architecture Parameters
        # ==========================
        self.input_channels = 25   # Number of features in PSM dataset
        self.kernel_size = 8       # Convolution kernel size, set smaller for Window=100
        self.stride = 1            # Stride
        self.final_out_channels = 128 # Feature depth of encoder output

        self.num_classes = 2       # 0: normal, 1: anomaly
        self.dropout = 0.35
        
        # features_len is the temporal length of features after CNN output
        # Input WIN_SIZE -> After 3 CNN layers + 3 MaxPool layers -> Output length
        # WIN_SIZE=100 -> features_len≈15, WIN_SIZE=256 -> features_len≈34
        # Note: This is just a default value, actual value should match the trained model
        self.features_len = 15    

        # ==========================
        # 2. Training Parameters
        # ==========================
        self.num_epoch = 40        # Number of training epochs, 40 epochs should be sufficient for PSM convergence

        # ==========================
        # 3. Optimizer Parameters
        # ==========================
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4             # Learning rate

        # ==========================
        # 4. Data Loading Parameters
        # ==========================
        self.drop_last = True      # Drop the last batch if it's smaller than batch_size
        self.batch_size = 32       # Recommended: Start with 32 if GPU memory is limited; reduce to 16 if needed

        # ==========================
        # 5. Sub-module Configurations
        # (These must not be deleted!)
        # ==========================
        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()


# ==========================================
# Data Augmentation Configuration
# ==========================================
class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 1.1 # Scaling strength parameter
        self.jitter_ratio = 0.8       # Noise addition parameter
        self.max_seg = 8              # Maximum number of segments for permutation


# ==========================================
# Contextual Contrasting Configuration
# ==========================================
class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.1          # Temperature parameter for InfoNCE Loss
        self.use_cosine_similarity = True # Use cosine similarity


# ==========================================
# Temporal Contrasting Configuration
# ==========================================
class TC(object):
    def __init__(self):
        self.hidden_dim = 100   # Hidden layer dimension inside TC module
        self.timesteps = 6      # Number of future timesteps to predict (cannot exceed features_len)
                                # Recommended to set smaller (e.g., 6-10) since features_len is only 15