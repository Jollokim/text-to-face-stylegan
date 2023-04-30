import os

import torch
from torch.utils.data import Dataset
import cv2 as cv

import pandas as pd
import numpy as np

from torchvision.transforms import transforms
from PIL import Image




class SimpleDataset(Dataset):
    def __init__(self, data_root, image_size) -> torch.tensor:

        self.root = data_root
        self.images = os.listdir(data_root)
        self.image_size = image_size

        self.transform = transforms.Compose([
            # Resize the image
            transforms.Resize(image_size),
            # Convert to PyTorch tensor
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.images[index])
        image = Image.open(img_path)


        # Transformation
        image = self.transform(image)
        return image.float()

    def __len__(self):
        return len(self.images)
    

