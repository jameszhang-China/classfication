import torch
import os
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T

DEFAULT_MEAN=[0.0, 0.0, 0.0], 
DEFAULT_STD=[1.0, 1.0, 1.0]  

class Dataset:
    def __init__(self, args):
        self.args = args

    def get_dataloader(self, path, batch_size, mode):
        dataset = self.build_dataset(path, mode)
        shuffle = mode == "train"
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.args.workers,
            pin_memory=torch.cuda.is_available()
        )
        return loader       

    def build_dataset(self, path, mode):
        base = ImageFolder(os.path.join(path, mode))
        transform = self.get_transforms(augment=(mode == 'train'))
        return ImageFolder(os.path.join(path, mode), transform=transform)

    def get_transforms(self, augment=False):
        if augment:
            self.transform = self.classify_augmentations(
                size = self.args.imgsz,
                mean = self.args.mean,
                std = self.args.std,
                scale = self.args.scale,
                hflip = self.args.fliplr,
                vflip = self.args.flipup,
                auto_augment = self.args.auto_augment,
                hsv_enabled = self.args.hsv_enabled,
                hsv_h = self.args.hsv_h,
                hsv_s = self.args.hsv_s,
                hsv_v = self.args.hsv_v
            )
        else: self.transform = self.classify_transforms(size = self.args.imgsz, mean=self.args.mean, std=self.args.std)
        return self.transform
    
    def classify_transforms(self,
        size: int = 224,
        mean: tuple[float, float, float] = DEFAULT_MEAN,
        std: tuple[float, float, float] = DEFAULT_STD,
        interpolation: str = "BILINEAR",
    ):
        scale_size = size if isinstance(size, (tuple, list)) and len(size) == 2 else (size, size)

        tfl = [T.Resize(scale_size[0], interpolation=getattr(T.InterpolationMode, interpolation))]
        tfl += [T.CenterCrop(size), T.ToTensor(), T.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))]
        return T.Compose(tfl)       

    def classify_augmentations(self,
        size: int = 224,
        mean: tuple[float, float, float] = DEFAULT_MEAN,
        std: tuple[float, float, float] = DEFAULT_STD,
        scale: tuple[float, float] = None,
        ratio: tuple[float, float] = None,
        hflip: float = 0.5,
        vflip: float = 0.0,
        auto_augment: bool = False,
        hsv_enabled: bool = False,
        hsv_h: float = 0.015,  # image HSV-Hue augmentation (fraction)
        hsv_s: float = 0.4,  # image HSV-Saturation augmentation (fraction)
        hsv_v: float = 0.4,  # image HSV-Value augmentation (fraction)
        erasing: float = 0.0,
        interpolation: str = "BILINEAR",
    ):
        scale = tuple(scale or (0.08, 1.0))  # default imagenet scale range
        ratio = tuple(ratio or (3.0 / 4.0, 4.0 / 3.0))  # default imagenet ratio range
        interpolation = getattr(T.InterpolationMode, interpolation)
        primary_tfl = [T.RandomResizedCrop(size, scale=scale, ratio=ratio, interpolation=interpolation)]
        if hflip > 0.0:
            primary_tfl.append(T.RandomHorizontalFlip(p=hflip))
        if vflip > 0.0:
            primary_tfl.append(T.RandomVerticalFlip(p=vflip))

        secondary_tfl = []

        if auto_augment:
            secondary_tfl.append(T.RandAugment(interpolation=interpolation))               
        if hsv_enabled:
            secondary_tfl.append(T.ColorJitter(brightness=hsv_v, contrast=hsv_v, saturation=hsv_s, hue=hsv_h))

        final_tfl = [
            T.ToTensor(),
            T.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
            T.RandomErasing(p=erasing, inplace=True),
        ]

        return T.Compose(primary_tfl + secondary_tfl + final_tfl)