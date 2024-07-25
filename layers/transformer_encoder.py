from torch import nn
from .multihead_attention import Multihead_Attention
from .mlp import MLP

class TransformerEncoderBlock(nn.Module):
    #2. Khởi tạo lớp với các tham số siêu từ Bảng 1 và Bảng 3 của bài báo ViT cho mô hình ViT-Base. 
    def __init__(self,embedding_dim:int=768,num_heads:int=12,mlp_size:int=3072,mlp_dropout:float=0.1,attn_dropout:float=0):
        super().__init__()
        #3. Khởi tạo một khối MSA
        self.msa_block=Multihead_Attention(dim=embedding_dim,num_heads=num_heads,attn_drop=attn_dropout)
        #4.Khởi tạo một khối MLP
        self.mlp_block=MLP(embedding_dim=embedding_dim,mlp_size=mlp_size,dropout_por=mlp_dropout)
    
    #5. Tạo 1 phương thức forward
    def forward(self,x):
        #6. Tạo kết nối bỏ qua cho khối MSA Block(Thêm Input vào output)
        x=self.msa_block(x)+x
        #7. Tạo kết nối bỏ qua cho khối MLP(Thêm Input vào output)
        x=self.mlp_block(x)+x
        return x
