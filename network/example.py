import torch 
from vision_transformer import SwinUnet

unet = SwinUnet(
    dim = 64,
    channels = 3,
    dim_mults = (1, 2, 4, 8),
    nested_unet_depths = (7, 4, 2, 1),     # nested unet depths, from unet-squared paper
    consolidate_upsample_fmaps = True,     # whether to consolidate outputs from all upsample blocks, used in unet-squared paper
)
unet.to('cuda')
img = torch.randn(1,3,512,512)
img = img.to('cuda')
out = unet(img) # (1, 3, 256, 256)
print(out.shape)

model_parameters = filter(lambda p: p.requires_grad, unet.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params)