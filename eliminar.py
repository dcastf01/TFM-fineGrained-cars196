##test

from timm.data import IMAGENET_DEFAULT_MEAN,IMAGENET_DEFAULT_STD
from timm.models.vision_transformer import _create_vision_transformer,vit_base_patch16_224_in21k
import timm
# cosas=dict(
#     img_size=448
#     )
# model=vit_base_patch16_224_in21k(pretrained=True,**cosas)

model_name="vit_base_patch16_224_in21k"
extras=dict(
    img_size=448
    )
model=timm.create_model(model_name,pretrained=True,**extras)
print(model)
# def _cfg(url='', **kwargs):
#     return {
#         'url': url,
#         'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
#         'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
#         'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
#         'first_conv': 'patch_embed.proj', 'classifier': 'head',
#         **kwargs
#     }
    
# default_cfgs ={
# 'vit_base_patch16_224_in21k_testeo': _cfg(
#         url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth',
        
#         input_size=(3,448,448),
#         num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

#         }


# def vit_base_patch16_224_in21k_testeo(pretrained=False, **kwargs):
#     """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
#     ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
#     """
#     model_kwargs = dict(
#         patch_size=16, embed_dim=768, depth=12, num_heads=12, representation_size=768, **kwargs)
#     model = _create_vision_transformer('vit_base_patch16_224_in21k_testeo', pretrained=pretrained, **model_kwargs)
#     return model
