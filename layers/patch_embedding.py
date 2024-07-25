from torch import nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels:int = 3, patch_size:int = 16, embedding_dim:int = 768):
        super().__init__()
        self.conv = nn.Conv2d(in_channels = in_channels, out_channels = embedding_dim, kernel_size = patch_size, padding = 0, stride = patch_size)

        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = x.transpose(-2,-1)

        return x
