"""
Script to perform data augmentation.
""" 
import os
import numpy as np
from data import RAVIRDataset
import torch 
import torchvision.transforms.functional as TF
from torchvision import transforms 
import matplotlib.pyplot as plt


# Define output directory for augmented images and masks
augmented_images_directory = "/media/axelrom16/HDD/AI/RAVIR_Project/data/train/images_augmented"
augmented_masks_directory = "/media/axelrom16/HDD/AI/RAVIR_Project/data/train/masks_augmented"

# Create the output directory if it doesn't exist
os.makedirs(augmented_images_directory, exist_ok=True)
os.makedirs(augmented_masks_directory, exist_ok=True)

# Create a custom dataset without data augmentation
train_data = RAVIRDataset(
    data_root='/media/axelrom16/HDD/AI/RAVIR_Project/data/train/training_images',
    segmentation_root='/media/axelrom16/HDD/AI/RAVIR_Project/data/train/training_masks',
    size=256,
    interpolation="bicubic",
    n_labels=3
)

# Define data augmentation transformations
def hflip(image, mask):
    return TF.hflip(image), TF.hflip(mask)

def vflip(image, mask):
    return TF.vflip(image), TF.vflip(mask)

def rotate(image, mask):
    params = transforms.RandomRotation.get_params((-45, 45))
    return TF.rotate(image, params), TF.rotate(mask, params)

def color_jitter(image, mask):
    return transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)(image), mask

def resized_crop(image, mask):
    params = transforms.RandomResizedCrop.get_params(image, scale=(0.8, 1.0), ratio=(0.75, 1.3))
    return TF.resized_crop(image, size=(256, 256), *params), TF.resized_crop(mask, size=(256, 256), *params)

def affine(image, mask):
    params = transforms.RandomAffine.get_params((-30, 30), (0.1, 0.1), (0.8, 1.2), None, (256, 256))
    return TF.affine(image, *params), TF.affine(mask, *params)

def perspective(image, mask):
    params = transforms.RandomPerspective.get_params(256, 256, 0.5)
    return TF.perspective(image, *params), TF.perspective(mask, *params)

def grayscale(image, mask):
    return transforms.RandomGrayscale(p=0.1)(image), mask

def erase(image, mask):
    params = transforms.RandomErasing.get_params(image, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=[0])
    mask = torch.cat((mask, mask, mask), dim=0)
    return TF.erase(image, *params), TF.erase(mask, *params)[0, ...].unsqueeze(0)

transformations = [
    hflip,
    vflip,
    rotate,
    color_jitter,
    resized_crop,
    affine,
    perspective,
    grayscale,
    erase
]
"""
transformations = [
    transforms.RandomHorizontalFlip(p=1),
    transforms.RandomVerticalFlip(p=1),
    transforms.RandomRotation(degrees=45),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0)),
    transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2)),
    transforms.RandomPerspective(),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1), ratio=(0.3, 3.3))
]
"""


skip_mask = [3, 7]
for i in range(len(train_data)):

    sample = train_data[i]
    image_name = sample['image_name']
    image = sample['image']
    image = torch.from_numpy(image).permute(2, 0, 1)
    #mask = (sample['mask'] * 255) / 2
    mask = sample['mask']
    mask = torch.from_numpy(mask).permute(2, 0, 1)

    image_orig = transforms.ToPILImage()(image)
    mask_orig = transforms.ToPILImage()(mask)

    image_orig.save(os.path.join(augmented_images_directory, f"{image_name}.png"))
    #mask_orig = (mask_orig / 255) * 2
    mask_orig.save(os.path.join(augmented_masks_directory, f"{image_name}.png"))

    for j, transformation in enumerate(transformations):

        """
        if j in skip_mask:
            mask_tr = mask
            image_tr = transformation(image)
        else:
            params = transformation.get_params()
        """
        image_tr, mask_tr = transformation(image, mask)
        
        image_tr = transforms.ToPILImage()(image_tr)
        mask_tr = transforms.ToPILImage()(mask_tr)

        # Save augmented pairs
        image_tr.save(os.path.join(augmented_images_directory, f"{image_name}_{j}_{i}.png"))
        #mask_tr = (mask_tr / 255) * 2
        mask_tr.save(os.path.join(augmented_masks_directory, f"{image_name}_{j}_{i}.png"))