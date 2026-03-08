import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class ColorDataset(Dataset):

    def __init__(self, folder):

        # Collect valid image paths
        self.paths = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):

        # Read image
        img = cv2.imread(self.paths[idx])

        # If image failed to load, skip safely
        if img is None:
            raise ValueError(f"Could not read image: {self.paths[idx]}")

        # Resize
        img = cv2.resize(img, (256, 256))

        # Convert BGR → LAB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        # Split channels
        L = img[:, :, 0].astype(np.float32) / 255.0
        AB = img[:, :, 1:].astype(np.float32) / 128.0 - 1

        # Convert to tensors
        L = torch.from_numpy(L).unsqueeze(0)
        AB = torch.from_numpy(AB).permute(2, 0, 1)

        return L, AB