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
import matplotlib.pyplot as plt
import albumentations
import cv2

    
class RAVIRDataset(Dataset):
    def __init__(self,
                 data_root,
                 segmentation_root,
                 size=None,
                 interpolation="bicubic",
                 n_labels=1
                 ):
        
        self.n_labels = n_labels
        self.data_root = data_root
        self.segmentation_root = segmentation_root
        
        self.image_ids = os.listdir(self.data_root)
        self.image_ids = [l.split('.')[0] for l in self.image_ids]
             
        self._length = len(self.image_ids)
        self.labels = {
            "image_name": self.image_ids,
            "relative_file_path_": [l+'.png' for l in self.image_ids],
            "file_path_": [os.path.join(self.data_root, l+'.png')
                           for l in self.image_ids],
            "segmentation_path_": [os.path.join(self.segmentation_root, l+'.png')
                                   for l in self.image_ids]
        }

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

        segmentation = Image.open(example["segmentation_path_"])
        segmentation = np.array(segmentation).astype(np.uint8)
        #segmentation = ((segmentation/255) * 2).astype(np.uint8)

        if self.size is not None:
            image = self.image_rescaler(image=image)["image"]
            segmentation = self.segmentation_rescaler(image=segmentation)["image"]
            processed = {"image": image,
                         "mask": segmentation}
        else:
            processed = {"image": image,
                         "mask": segmentation}

        example["image"] = (processed["image"]/255).astype(np.float32)

        example["mask"] = np.expand_dims(processed["mask"], -1).astype(np.uint8)

        segmentation = processed["mask"]
        onehot = np.eye(self.n_labels)[segmentation].astype(np.float32)
        example["segmentation"] = onehot

        s_onehot = ((onehot/np.max(onehot))*2)-1
        example["imageseg"] = np.concatenate((example["image"], s_onehot), axis=2)

        example['aesegmentation'] = s_onehot

        return example

