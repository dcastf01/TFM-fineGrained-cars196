import albumentations as A
from albumentations.pytorch import ToTensorV2
from timm.data.constants import IMAGENET_DEFAULT_MEAN,IMAGENET_DEFAULT_STD
from config import CONFIG
basic_transform=A.Compose(
        [
            A.Resize(CONFIG.IMG_SIZE,CONFIG.IMG_SIZE),
            A.Normalize(
                # mean=[0, 0, 0],
                mean=[IMAGENET_DEFAULT_MEAN[0], IMAGENET_DEFAULT_MEAN[1], IMAGENET_DEFAULT_MEAN[2]],
                # std=[1, 1, 1],
                std=[IMAGENET_DEFAULT_STD[0], IMAGENET_DEFAULT_STD[1], IMAGENET_DEFAULT_STD[2]],
                max_pixel_value=255,
                ),
            ToTensorV2(),
                ]
                
                )