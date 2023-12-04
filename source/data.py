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


class RAVIRDataset(Dataset):
    """
    Dataset class for the RAVIR dataset
    """
    def __init__(self,
                 root_dir,
                 train=True):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.train = train

        if self.train:
            self.images = os.listdir(os.path.join(self.root_dir, 'training_images'))
            self.masks = os.listdir(os.path.join(self.root_dir, 'training_masks'))
        else:
            self.images = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_path)
        image = image.convert('RGB')

        mask = None
        if self.train:
            mask_path = os.path.join(self.root_dir, self.masks[idx])
            mask = Image.open(mask_path)

        return image, mask
    
class SegmentationBase(Dataset):
    def __init__(self,
                 data_csv, data_root, segmentation_root,
                 size=None, random_crop=False, interpolation="bicubic",
                 n_labels=1, shift_segmentation=False,
                 augmentation=False,
                 ):
        self.n_labels = n_labels
        self.shift_segmentation = shift_segmentation
        self.data_csv = data_csv
        self.data_root = data_root
        self.segmentation_root = segmentation_root
        
        self.image_ids = []
        self.gleason_list = []
        with open(os.path.join(self.data_root, self.data_csv), "r") as f:
            lines = f.read().splitlines()
            for line in lines[1:]:
                self.image_ids.append(line.split(',')[1])
                self.gleason_list.append(line.split(',')[2])
             
        self._length = len(self.image_ids)
        self.labels = {
            "image_name": self.image_ids,
            "relative_file_path_": [l+'.png' for l in self.image_ids],
            "file_path_": [os.path.join(self.data_root, l+'.png')
                           for l in self.image_ids],
            "segmentation_path_": [os.path.join(self.segmentation_root, l+'_mask.png')
                                   for l in self.image_ids],
            "gleason": self.gleason_list
        }

        size = None if size is not None and size<=0 else size
        self.size = size

        if augmentation:
            self.augmentation = albumentations.OneOf(
                    [albumentations.VerticalFlip(p=0.25),
                    albumentations.HorizontalFlip(p=0.25),
                     albumentations.RandomRotate90(p=0.25)
                    ])
        else:
            self.augmentation = None

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
            self.center_crop = not random_crop
            if self.center_crop:
                self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size, width=self.size)
            self.preprocessor = self.cropper

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

        if self.size is not None:
            image = self.image_rescaler(image=image)["image"]

        if self.shift_segmentation:
            # used to support segmentations containing unlabeled==255 label
            segmentation = segmentation+1
        if self.size is not None:
            segmentation = self.segmentation_rescaler(image=segmentation)["image"]
        if self.size is not None:
            processed = self.preprocessor(image=image,
                                          mask=segmentation
                                          )
        else:
            processed = {"image": image,
                         "mask": segmentation
                         }

        if self.augmentation is not None:
            processed = self.augmentation(image=processed['image'], mask=processed['mask'])       

        example["image"] = (processed["image"]/127.5 - 1.0).astype(np.float32)

        example["mask"] = np.expand_dims(processed["mask"], -1)

        segmentation = processed["mask"]
        onehot = np.eye(self.n_labels)[segmentation].astype(np.float32)
        example["segmentation"] = onehot

        s_onehot = ((onehot/np.max(onehot))*2)-1
        example["imageseg"] = np.concatenate((example["image"], s_onehot), axis=2)

        example['aesegmentation'] = s_onehot

        return example




if __name__ == '__main__':
    data = RAVIRDataset(root_dir='data/train',
                        train=True)
    print(len(data))

    image, mask = data[0]
    print(image.size)
    print(mask.size)

    plt.imshow(image)
    plt.show()
    plt.imshow(mask)
    plt.show()