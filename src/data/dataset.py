import os
import re
import warnings
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import cv2
import numpy as np
import tifffile as tif
from PIL import Image
import skimage as sci

import torch
from torchvision.ops import masks_to_boxes
from torchvision.datasets import VisionDataset

def create_dataset_folder(root: str, folder: str, val_files: list, img_format: str = 'tif'):
    root = Path(root)
    folder = Path(folder)

    for _dir in ('images', 'masks'):
        Path(folder / 'train' / _dir).mkdir(parents=True, exist_ok=True)
        Path(folder / 'val' / _dir).mkdir(parents=True, exist_ok=True)

    for file in os.listdir(root):
        if 'Mask' in file:
            continue

        sub_folder = 'val' if file in val_files else 'train'
        mask_path = root / file.replace('.tif', '_Mask.tif')
        if not mask_path.is_file():
            warnings.warn(f'No mask for {file}', category=UserWarning)
            continue

        img = tif.imread(root / file)[2]

        i, j = re.findall(r'\d+', file)
        i = '0' * (3 - len(i)) + i
        img_path = folder / sub_folder / 'images' / f'fused_s{i}_{j}.{img_format}'
        cv2.imwrite(str(img_path), img)
        mask = tif.imread(mask_path)[0]
        mask_path = folder / sub_folder / 'masks' / f'fused_s{i}_{j}.{img_format}'
        cv2.imwrite(str(mask_path), mask)

def masks2bboxes(masks: np.array) -> np.array:
    bboxes = np.zeros((masks.shape[0], 4), dtype=int)

    for index, mask in enumerate(masks):
        y, x = np.where(mask != 0)

        bboxes[index, 0] = np.min(x)
        bboxes[index, 1] = np.min(y)
        bboxes[index, 2] = np.max(x) - np.min(x)
        bboxes[index, 3] = np.max(y) - np.min(y)

    return bboxes

def square_crop(img, bbox, square_side):
    xmin, ymin, width, height = bbox
    crop_bbox = img[ymin:ymin + height, xmin:xmin + width]

    pad_top = (square_side - height) // 2
    pad_bottom = square_side - height - pad_top
    pad_left = (square_side - width) // 2
    pad_right = square_side - width - pad_left
    pad_width = ((pad_top + 20, pad_bottom + 20), (pad_left + 20, pad_right + 20))

    return np.pad(crop_bbox, pad_width)

class Nuclei(VisionDataset):
    def __init__(
            self,
            image_paths: list,
            transforms: Optional[Callable] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(image_paths, transforms, transform, target_transform)
        self.images, self.masks = list(), list()
        bbox_max_side = 0
        for image_path in image_paths:
            images = sci.io.imread(image_path)
            masks_tubules = sci.io.imread(image_path.replace(".tif", "_Mask.tif"))
            masks_nuclei = sci.io.imread(image_path.replace(".tif", "_nuclei.tif"))

            masks_tubules_ids = np.unique(masks_tubules)[1:254]
            masks_tubules = masks_tubules == masks_tubules_ids[:, None, None]
            images = images * masks_tubules
            masks_nuclei = masks_nuclei * masks_tubules
            bboxes = masks2bboxes(masks_tubules)
            bbox_max_side = max(bbox_max_side, np.max(bboxes[:, 2:]))
            for bbox, img, nuclei in zip(bboxes, images, masks_nuclei):
                square_img = square_crop(img, bbox, bbox_max_side)
                square_nuclei = square_crop(nuclei, bbox, bbox_max_side)
                self.images.append(Image.fromarray(np.uint8(square_img)).convert('RGB'))
                self.masks.append(Image.fromarray(np.uint8(square_nuclei)).convert('RGB'))


    def __getitem__(self, index: int) -> tuple[Any, Any]:
        img = self.images[index]
        mask = self.masks[index]

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        single_channel_mask = mask > 1
        return img[0].unsqueeze(0).float(), single_channel_mask[0].unsqueeze(0).float()


    def __len__(self) -> int:
        return len(self.images)


class Tubules(VisionDataset):

    def __init__(
            self,
            root: str,
            transforms: Optional[Callable] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self.root = Path(root)
        self.images = [Image.open(file) for file in Path(self.root / 'images').iterdir()]
        self.masks = [Image.open(file) for file in Path(self.root / 'masks').iterdir()]

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        img = self.images[index]
        mask = self.masks[index]

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self) -> int:
        return len(self.images)


class UnetTubules(Tubules):

    def __init__(
            self,
            root: str,
            transforms: Optional[Callable] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        img, mask = super().__getitem__(index)  # Call parent class method
        single_channel_mask = mask > 1
        return img.float(), single_channel_mask.float()


class SamTubules(VisionDataset):
    """Tubules Segmentation Dataset for segment anything model"""

    def __init__(
            self,
            root: str,
            transforms: Optional[Callable] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self.root = Path(root)
        self.images = [Image.open(file) for file in Path(self.root / 'images').iterdir()]
        self.masks = [Image.open(file) for file in Path(self.root / 'masks').iterdir()]

    def __getitem__(self, index: int) -> tuple[Any, Any, Any]:
        img = self.images[index]
        mask = self.masks[index]

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        # bboxes
        instance_ids = torch.unique(mask)[1:]
        instance_masks = mask == instance_ids[:, None, None]
        bboxes = masks_to_boxes(instance_masks)

        return img, mask, bboxes

    def __len__(self) -> int:
        return len(self.images)
