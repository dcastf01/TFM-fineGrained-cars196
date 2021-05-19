import albumentations as A
from torchvision import transforms
from albumentations.pytorch import ToTensorV2
from timm.data.constants import IMAGENET_DEFAULT_MEAN,IMAGENET_DEFAULT_STD
from timm.data.transforms_factory import transforms_noaug_train,transforms_imagenet_train,transforms_imagenet_eval
from PIL import Image
from autoaugment import AutoAugImageNetPolicy

def cars_train_transfroms_transFG(img_size:int=448):
    transforms=transforms.Compose([
                                    transforms.Resize((600, 600), Image.BILINEAR),
                                    transforms.RandomCrop((448, 448)),
                                    transforms.RandomHorizontalFlip(),
                                    AutoAugImageNetPolicy(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
                       )                            
    return transforms
    
def cars_test_transfroms_transFG(img_size:int=448):
    transform=transforms.Compose([
                                    transforms.Resize((600, 600), Image.BILINEAR),
                                    transforms.CenterCrop((448, 448)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
                            )                 
    return transforms

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
