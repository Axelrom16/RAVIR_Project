"""
Script for the definition of the datasets. 
""" 
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import albumentations
import cv2
import random


def increase_contrast(image, alpha=1.5, beta=0):
    """
    Apply contrast enhancement to the input image.

    Parameters:
        image (numpy.ndarray): Input image.
        alpha (float): Contrast control (1.0 means no change).
        beta (float): Brightness control.

    Returns:
        numpy.ndarray: Image with increased contrast.
    """
    enhanced_image = np.clip(alpha * image + beta, 0, 255).astype(np.uint8)
    return enhanced_image


def apply_histogram_equalization(image):
    """
    Apply histogram equalization to the input image.

    Parameters:
        image (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Image with histogram equalization applied.
    """

    # Convert the image to grayscale if it's a color image
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Ensure that the image is of type CV_8UC1 (8-bit unsigned single-channel)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    # Apply adaptive histogram equalization (CLAHE)
    equalized_image = cv2.equalizeHist(image)

    return equalized_image


class ContrastEnhancementTransform(object):
    def __init__(self, alpha=1.5, beta=0):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, sample):
        # Assume sample is a PIL Image, you may need to adjust accordingly
        enhanced_image = increase_contrast(np.array(sample), alpha=self.alpha, beta=self.beta)
        return enhanced_image


class HistogramEqualizationTransform(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        # Assume sample is a PIL Image, you may need to adjust accordingly
        enhanced_image = apply_histogram_equalization(np.array(sample))
        return enhanced_image
    

class RAVIRDataset(Dataset):
    def __init__(self,
                 data_root,
                 segmentation_root,
                 size=None,
                 interpolation="bicubic",
                 n_labels=1,
                 augmentation=False,
                 contrast_enhancement=False,
                 ):
        
        self.n_labels = n_labels
        self.data_root = data_root
        self.segmentation_root = segmentation_root
        self.augmentation = augmentation    
        self.contrast_enhancement = contrast_enhancement
        
        self.image_ids = os.listdir(self.data_root)
        self.image_ids = [l.split('.')[0] for l in self.image_ids]
             
        self._length = len(self.image_ids)

        if self.segmentation_root is None:
            self.labels = {
                "image_name": self.image_ids,
                "relative_file_path_": [l+'.png' for l in self.image_ids],
                "file_path_": [os.path.join(self.data_root, l+'.png')
                            for l in self.image_ids]
            }
        else:
            self.labels = {
                "image_name": self.image_ids,
                "relative_file_path_": [l+'.png' for l in self.image_ids],
                "file_path_": [os.path.join(self.data_root, l+'.png')
                            for l in self.image_ids],
                "segmentation_path_": [os.path.join(self.segmentation_root, l+'.png')
                                    for l in self.image_ids]
            }

        if augmentation:
            self.augmentation = albumentations.OneOf(
                    [albumentations.VerticalFlip(p=1),
                     albumentations.HorizontalFlip(p=1),
                     albumentations.Perspective(p=1, scale=(0.05, 0.1)),
                     albumentations.Affine(p=1, scale=(0.75, 1.25), shear=15),
                     albumentations.PiecewiseAffine(p=1, scale=(0.03, 0.05))
                    ])
        else:
            self.augmentation = None

        self.contrast_enhancement = HistogramEqualizationTransform() if contrast_enhancement else None

        size = None if size is not None and size<=0 else size
        self.size = size

        if self.size is not None:
            self.interpolation = interpolation
            self.interpolation = {
                "nearest": cv2.INTER_NEAREST,
                "bilinear": cv2.INTER_LINEAR,
                "bicubic": cv2.INTER_CUBIC,
                "area": cv2.INTER_AREA,
                "lanczos": cv2.INTER_LANCZOS4}[self.interpolation]
            self.image_rescaler = albumentations.SmallestMaxSize(max_size=self.size,
                                                                 interpolation=self.interpolation)
            self.segmentation_rescaler = albumentations.SmallestMaxSize(max_size=self.size,
                                                                        interpolation=cv2.INTER_NEAREST)

    def __len__(self):
        return self._length
    
    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)

        if self.segmentation_root is None:
            segmentation = None
        else:
            segmentation = Image.open(example["segmentation_path_"])
            segmentation = np.array(segmentation).astype(np.uint8)
            segmentation = np.expand_dims(segmentation, -1)

        if self.size is not None:
            image = self.image_rescaler(image=image)["image"]
            if segmentation is not None:
                segmentation = self.segmentation_rescaler(image=segmentation)["image"]
                processed = {"image": image,
                            "mask": segmentation}
            else:
                processed = {"image": image}
        else:
            if segmentation is not None:
                processed = {"image": image,
                            "mask": segmentation}
            else:    
                processed = {"image": image}

        if self.augmentation is not None:
            if segmentation is not None:
                processed = self.augmentation(image=processed['image'], mask=processed['mask'])
            else:  
                processed = self.augmentation(image=processed['image'])

        if self.contrast_enhancement is not None:
            processed['image'] = processed['image'][..., 0] 
            processed['image'] = processed['image'].astype(np.uint8)
            equalized_image = self.contrast_enhancement(processed['image'])
            equalized_image = np.concatenate((equalized_image[..., np.newaxis], equalized_image[..., np.newaxis], equalized_image[..., np.newaxis]), axis=2)
            processed['image'] = equalized_image
        
        example["image"] = (processed["image"]/255).astype(np.float32)
        example["image"] = torch.from_numpy(example["image"])

        if segmentation is None:
            return example
        
        example["mask"] = ((processed["mask"]/255)*2).astype(np.uint8)
        segmentation = example["mask"]
        onehot = np.eye(self.n_labels)[segmentation].astype(np.float32)
        example["segmentation"] = onehot[..., 0, :]
        example["mask"] = torch.from_numpy(example["mask"])
        example["segmentation"] = torch.from_numpy(example["segmentation"])

        return example
    
    
"""
if __name__ == "__main__":
    
    data = RAVIRDataset(
        data_root='/media/axelrom16/HDD/AI/RAVIR_Project/data/train/images_augmented',
        segmentation_root='/media/axelrom16/HDD/AI/RAVIR_Project/data/train/masks_augmented',
        size=768,
        interpolation="bicubic",
        n_labels=3,
        augmentation=True
    )

    ex = data[0]

    print(ex['image'].shape)
    print(ex['image'].min(), ex['image'].max())
    print(ex['mask'].shape)
    print(ex['mask'].min(), ex['mask'].max())
    print(ex['segmentation'].shape)
    print(ex['segmentation'].min(), ex['segmentation'].max())

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(ex['image'])
    axs[1].imshow(ex['mask'], cmap='gray')
    plt.show()
"""