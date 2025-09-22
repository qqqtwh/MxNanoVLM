import torch
import math
from torchvision.transforms.functional import resize,InterpolationMode
from einops import rearrange
from typing import Tuple,Union
from PIL import Image

class DynamicResize(torch.nn.Module):
    ''' 将图像缩放到一个新的尺寸，能被patch_size整除，同时不超过max_side_len
        p=512,m=1024,allow_upscale=True
        _get_new_hw(300,400)   # 512,512
        _get_new_hw(800,1200)  # 1024,1024
        _get_new_hw(400,300)   # 512,512
        _get_new_hw(1200,800)  # 1024,1024
        
        p=512,m=1024,allow_upscale=False
        _get_new_hw(300,400)   # 512,400
        _get_new_hw(800,1200)  # 1024,1024
        _get_new_hw(400,300)   # 400,512
        _get_new_hw(1200,800)  # 1024,1024
    '''

    def __init__(
        self,
        patch_size: int,    # vit_img_size
        max_side_len: int,  # max_img_size
        Interpolation: InterpolationMode = InterpolationMode.BICUBIC,
        allow_upscale: bool = True
    ) -> None:
        super().__init__()
        self.p = int(patch_size)
        self.m = int(max_side_len)
        self.interpolation = Interpolation
        self.allow_upscale = allow_upscale
    
    def _get_new_hw(self,h:int, w:int)->Tuple[int,int]:
        # 获得可以被 patch_size 整除的新hw

        long,short = (w,h) if w>=h else (h,w)

        target_long = min(self.m, math.ceil(long/self.p)*self.p)
        if not self.allow_upscale:
            target_long = min(target_long,long)

        scale = target_long/long

        target_short = math.ceil(short*scale/self.p) * self.p
        target_short = max(target_short, self.p)

        return (target_short,target_long) if w>=h else (target_long,target_short)
            

    def forward(self, img:Union[Image.Image, torch.Tensor]):
        if isinstance(img,Image.Image):
            w,h = img.size
            new_h,new_w = self._get_new_hw(h,w)
            return resize(img, [new_h, new_w], interpolation=self.interpolation)
        
        if not torch.is_tensor(img):
            raise TypeError("DynamicResize expects a PIL Image or a torch.Tensor; " f"got {type(img)}")
    
        batched = img.dim==4
        if img.dim not in (3,4):
            raise TypeError("Tensor input must have shape (C,H,W) or (B,C,H,W); " f"got {img.shape}")
        
        imgs = img if batched else img.unsqueeze(0)
        _,_,h,w = imgs.shape
        new_h,new_w = self._get_new_hw(h,w)
        out = resize(imgs,[new_h,new_w],interpolation=self.interpolation)

        return out if batched else out.squeeze(0)
    
class SplitImage(torch.nn.Module):
    '''
        把图切成 patch_size x patch_size 的小块，堆叠成一个 patches:
            p=512时: (1,3,1024,1024) 
                patches --> (4,3,512,512)
                n_h, n_w --> 2,2

            原图 (1024x1024)
            +---------------------+
            |  patch1  |  patch2  |
            | (512x512)| (512x512)|
            +----------+----------+
            |  patch3  |  patch4  |
            | (512x512)| (512x512)|
            +---------------------+

            1张图 → 1×4 = 4个 patches
    '''

    def __init__(self, patch_size:int)->None:
        super().__init__()
        self.p = patch_size

    def forward(self, x:torch.Tensor) -> Tuple[torch.Tensor, Tuple[int,int]]:
        
        if x.ndim==3:
            x = x.unsqueeze(0)
        
        b,c,h,w = x.shape
        if h%self.p or w%self.p:
            raise ValueError(f'Image size {(h,w)} not divisible by patch_size {self.p}')
        
        n_h, n_w = h//self.p, w//self.p
        patches = rearrange(x, 'b c (nh ph) (nw pw) -> (b nh nw) c ph pw',ph=self.p, pw=self.p)

        return patches, (n_h,n_w)
    
    