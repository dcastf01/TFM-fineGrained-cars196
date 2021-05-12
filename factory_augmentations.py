import albumentations as A
from albumentations.pytorch import ToTensorV2
from timm.data.constants import IMAGENET_DEFAULT_MEAN,IMAGENET_DEFAULT_STD
from timm.data.transforms_factory import transforms_noaug_train,transforms_imagenet_train

from config import CONFIG

def basic_transforms(img_size:int=CONFIG.IMG_SIZE):
    basic_transform=A.Compose(
            [
                A.Resize(img_size,img_size),
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
    return basic_transform
