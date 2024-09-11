import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class EmbryoDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index].replace(".jpg",".BMP")) #load 1 channel image
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg",".BMP")) #load 1 channel mask
        image = np.array(Image.open(img_path).convert("RGB")) #make 3 channel image
        mask = np.array(Image.open(mask_path).convert("L")) #converting masks in to grayscale

        #preprocessing for the masks
        mask[mask == 255] = 4  #ICM
        mask[mask == 192] = 3  #Blastocoel
        mask[mask == 226] = 3 # Blastocoel
        mask[mask == 128] = 2  #TE
        mask[mask == 64] = 1    #ZP
        mask[mask == 105] = 1 # ZP
        mask[mask == 0] = 0    #Background

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        return image, mask



class HAMDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index]) #load 1 channel image
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg",".png"))  #load 1 channel mask
        image = np.array(Image.open(img_path).convert("RGB")) #make 3 channel image
        mask = np.array(Image.open(mask_path).convert("L")) #converting masks in to grayscale

        #preprocessing for the masks
        #mask[mask == 0] = 0   #Background
        mask[mask < 100] = 0  # Background
        mask[mask > 100] = 1  # tumor



        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        return image, mask