import albumentations as A
from torchvision import transforms
from albumentations.pytorch import ToTensorV2
from timm.data.constants import IMAGENET_DEFAULT_MEAN,IMAGENET_DEFAULT_STD
from timm.data.transforms_factory import transforms_noaug_train,transforms_imagenet_train,transforms_imagenet_eval
from PIL import Image
from autoaugment import AutoAugImageNetPolicy
from config import ModelsAvailable

def get_normalize_parameter_by_model(model_enum:ModelsAvailable):
    prefix_name=model_enum.name[0:3]
    if model_enum==ModelsAvailable.vit_base_patch16_224_miil_in21k:
        mean=(0,0,0)
        std=(1, 1, 1)
        interpolation=Image.BILINEAR
    elif model_enum.name.split("_")[-1]=="overlap":
        mean=IMAGENET_DEFAULT_MEAN
        std=IMAGENET_DEFAULT_STD
        interpolation=Image.BILINEAR
        
    elif prefix_name==ModelsAvailable.vit_large_patch16_224_in21k.name[0:3] \
        and model_enum!=ModelsAvailable.vit_base_patch16_224_miil_in21k:
        
        mean=(0.5,0.5,0.5)
        std=(0.5, 0.5, 0.5)
        interpolation=Image.BICUBIC

    else:
        mean=IMAGENET_DEFAULT_MEAN
        std=IMAGENET_DEFAULT_STD
        interpolation=Image.BICUBIC
    return mean,std,interpolation

def cars_train_transfroms_transFG(img_size:int=448,
                                  mean:list=IMAGENET_DEFAULT_MEAN,
                                  std:list=IMAGENET_DEFAULT_STD,
                                  interpolation=Image.BILINEAR,
                                  ):
    transform=transforms.Compose([
                                    transforms.Resize((600, 600), interpolation),
                                    transforms.RandomCrop((img_size, img_size)),
                                    transforms.RandomHorizontalFlip(),
                                    # AutoAugImageNetPolicy(),
                                    transforms.AutoAugment(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)]
                       )                            
    return transform
    
def cars_test_transfroms_transFG(img_size:int=448,
                                  mean:list=IMAGENET_DEFAULT_MEAN,
                                  std:list=IMAGENET_DEFAULT_STD,
                                  interpolation=Image.BILINEAR,
                                  ):
    transform=transforms.Compose([
                                    transforms.Resize((600, 600), interpolation),
                                    transforms.CenterCrop((img_size, img_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)]
                            )                 
    return transform

def basic_transforms(img_size:int=224):
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

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]
