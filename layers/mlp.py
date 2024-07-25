from torch import nn

class MLP(nn.Module):
    def __init__(self, embedding_dim:int = 768, mlp_size:int = 3072, dropout_por:float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape=embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p = dropout_por),
            nn.Linear(in_features=mlp_size , out_features=embedding_dim),
            nn.Dropout(p = dropout_por)
        )
    def forward(self, x):
        x = self.norm(x)
        x = self.mlp(x)

        return x