import os

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class GANDataset(Dataset):
    def __init__(self, picle_file: str, descriptions_folder: str, image_folder: str, description_encoder, transform=transforms.ToTensor()):
        self.names = np.load(picle_file, allow_pickle=True)
        self.description_folder = descriptions_folder
        self.image_folder = image_folder
        
        self.description_encoder = description_encoder
        self.transform = transform

    def __getitem__(self, index):
        name = self.names[index]

        descriptions = list(np.genfromtxt(os.path.join(self.description_folder, f'{name}.txt'), dtype=str, delimiter='\n'))
        img = cv.imread(os.path.join(self.image_folder, f'{name}.jpg'))
        img = self.transform(img)

        item = {'descriptions': descriptions, 'images': img}

        return item

    def __len__(self):
        return len(self.names)


# testing dataset class
def main():
    dataset = GANDataset(r'data/train.pickle', r'data/descriptions', r'data/images', None)

    data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            num_workers=8,
            drop_last=False,
            shuffle=True
        )

    item = dataset.__getitem__(0)

    print(item['descriptions'])
    print(item['images'].shape)

    for batch in data_loader:
        print(batch['images'].shape)
        print(len(batch['descriptions'][0]))

        quit()

if __name__ == '__main__':
    main()
