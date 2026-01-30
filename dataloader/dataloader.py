import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np
from .augmentations import DataTransform


class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset, config, training_mode):
        super(Load_Dataset, self).__init__()
        self.training_mode = training_mode

        X_train = dataset["samples"]
        y_train = dataset["labels"]

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
            X_train = X_train.permute(0, 2, 1)

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train

        # Check for NaN or Inf values in data
        if torch.isnan(self.x_data).any() or torch.isinf(self.x_data).any():
            print("Warning: Data contains NaN or Inf values!")
            # Replace NaN and Inf with 0
            self.x_data = torch.where(torch.isnan(self.x_data) | torch.isinf(self.x_data), 
                                     torch.zeros_like(self.x_data), self.x_data)

        self.len = X_train.shape[0]
        self.config = config

    def __getitem__(self, index):
        if self.training_mode == "self_supervised":
            # Apply random data augmentation each time __getitem__ is called, ensuring different augmentations each epoch
            aug1, aug2 = DataTransform(self.x_data[index], self.config)
            return self.x_data[index], self.y_data[index], aug1, aug2
        else:
            return self.x_data[index], self.y_data[index], self.x_data[index], self.x_data[index]

    def __len__(self):
        return self.len


def data_generator(data_path, configs, training_mode):
    # Check if data path exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path does not exist: {data_path}")
    
    # Check if required files exist
    train_file = os.path.join(data_path, "train.pt")
    val_file = os.path.join(data_path, "val.pt")
    test_file = os.path.join(data_path, "test.pt")
    
    for file_path, file_name in [(train_file, "train.pt"), (val_file, "val.pt"), (test_file, "test.pt")]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
    
    print(f"Loading data from {data_path}...")
    train_dataset = torch.load(train_file)
    valid_dataset = torch.load(val_file)
    test_dataset = torch.load(test_file)
    
    # Validate data format
    for name, dataset in [("train", train_dataset), ("val", valid_dataset), ("test", test_dataset)]:
        if not isinstance(dataset, dict) or "samples" not in dataset or "labels" not in dataset:
            raise ValueError(f"{name}.pt file format is incorrect, should contain 'samples' and 'labels' keys")
        print(f"  {name}.pt: samples shape={dataset['samples'].shape}, labels shape={dataset['labels'].shape}")

    train_dataset = Load_Dataset(train_dataset, configs, training_mode)
    valid_dataset = Load_Dataset(valid_dataset, configs, training_mode)
    test_dataset = Load_Dataset(test_dataset, configs, training_mode)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=configs.batch_size,
                                               shuffle=False, drop_last=configs.drop_last,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.batch_size,
                                              shuffle=False, drop_last=False,
                                              num_workers=0)

    return train_loader, valid_loader, test_loader