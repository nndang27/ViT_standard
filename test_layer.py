from layers import PatchEmbedding, Multihead_Attention
import torch
from modeling.vit_model import ViT
from torch import nn

# class Multihead_Attention2(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
#         if qk_scale is None:
#             self.scale = head_dim ** -0.5
#         else:
#             self.scale = qk_scale

#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.dim = dim

#     def forward(self, x):
#         B, N, C = x.shape
#         # print('for attention',x.shape)
#         # print(self.scale)
#         q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads,
#                                       C // self.num_heads).permute(2, 0, 3, 1, 4)
#         # print(q.shape,k.shape,v.shape)
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#         print("SIZE of attn map 3: ", attn[0,0,0,:10])
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x, attn

def main():
    image = torch.randn(1,3,224,224)
    vit = ViT('B_16_imagenet1k', pretrained=True, image_size=224)
    output = vit(image)
    print(output.shape)
    # for name, param in vit.named_parameters():
    #   print(name, "  ", param)
    return

if __name__ == "__main__":
    main()