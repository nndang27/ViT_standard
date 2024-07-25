from torch import nn

class Multihead_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        if qk_scale is None:
            self.scale = head_dim ** -0.5
        else:
            self.scale = qk_scale

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.dim = dim

    def forward(self, x):
        B, N, C = x.shape
        # print('for attention',x.shape)
        # print(self.scale)
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                      C // self.num_heads).permute(2, 0, 3, 1, 4)
        # print(q.shape,k.shape,v.shape)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class Multihead_Attention(nn.Module):
#     def __init__(self, embed_dim, num_heads):
#         super().__init__()
#         assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads

#         self.q_linear = nn.Linear(embed_dim, embed_dim)
#         self.k_linear = nn.Linear(embed_dim, embed_dim)
#         self.v_linear = nn.Linear(embed_dim, embed_dim)
#         self.out_linear = nn.Linear(embed_dim, embed_dim)

#     def scaled_dot_product_attention(self, q, k, v, mask=None):
#         scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
#         if mask is not None:
#             scores = scores.masked_fill(mask == 0, float('-inf'))

#         attn_weights = F.softmax(scores, dim=-1)
#         output = torch.matmul(attn_weights, v)
#         return output, attn_weights

#     def forward(self, query, key, value, mask=None):
#         batch_size = query.size(0)
        
#         # Linear projections
#         q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim)
#         k = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim)
#         v = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim)

#         q = q.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
#         k = k.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
#         v = v.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)

#         # Scaled dot-product attention
#         output, attn_weights = self.scaled_dot_product_attention(q, k, v, mask)

#         # Concatenate heads and apply final linear layer
#         output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
#         output = self.out_linear(output)
        
#         return output, attn_weights

