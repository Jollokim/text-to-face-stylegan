import os

import torch
from torch.utils.data import Dataset

import pandas as pd
import numpy as np

from torchvision.transforms import transforms
from PIL import Image




class T2FDataset(Dataset):
    def __init__(self, img_root, image_size, description_root) -> torch.tensor:

        self.root = img_root
        self.images = os.listdir(img_root)
        self.image_size = image_size
        self.description_root = description_root

        self.transform = transforms.Compose([
            # Resize the image
            transforms.Resize(image_size),
            # Convert to PyTorch tensor
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        
        # picking the description
        description_file = os.path.join(self.description_root, self.images[index][:self.images[index].index('.jpg')] + '.txt')

        with open(description_file, 'r') as file:
            descriptions = file.read().split('\n')

        rand_num = np.random.randint(0, len(descriptions))

        r_description = descriptions[rand_num]

        # picking the image
        img_path = os.path.join(self.root, self.images[index])
        image = Image.open(img_path)

        # Transformation
        image_transformed = self.transform(image)
        image = torch.tensor(np.asarray(Image.open(img_path)))
        return image, image_transformed.float(), r_description

    def __len__(self):
        return len(self.images)
    
if __name__ == '__main__':
    dataset = T2FDataset('data/images_test', 16, 'data/description_test')
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        num_workers=5,
        shuffle=True,
        drop_last=True, 
        pin_memory=True
    )

    img, description = dataset.__getitem__(0)

    print(description)

    for real, descriptions in dataloader:
        print(real.shape)
        print(descriptions.shape, descriptions.dtype)
        quit()
    

