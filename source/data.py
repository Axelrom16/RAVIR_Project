"""
Script for the definition of the datasets. 
""" 
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class RAVIRDataset(Dataset):
    """
    Dataset class for the RAVIR dataset
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.files = os.listdir(root_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.files[idx])
        image = Image.open(img_name)
        image = image.convert('RGB')
        label = int(self.files[idx][0:4]) - 1

        if self.transform:
            image = self.transform(image)

        return image, label