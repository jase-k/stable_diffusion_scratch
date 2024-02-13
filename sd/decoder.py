import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class VAE_AttentionBlock(nn.Module):

    def __init__(self, in_channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels) # Group Normalization: https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html
        self.attention = SelfAttention(1, in_channels) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #  x: (Batch_Size, Channel, Height, Width)
        residue = x

        n, c, h, w = x.shape()


        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * Width)
        x = x.view(n, c, h * w)

        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Height * Width, Features)
        x = x.transpose(-1, -2)

        # see attention.py for more details
        x = self.attention(x)

        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Features, Height * Width)
        x = x.transpose(-1, -2)

        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Features, Height, Width)
        x = x.view(n, c, h, w)

        return x + residue

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels) # Group Normalization: https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1) # Convusional Visualizer: https://ezyang.github.io/convolution-visualizer/

        self.groupnorm_2 = nn.GroupNorm(32, out_channels) # Group Normalization: https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) # Convusional Visualizer: https://ezyang.github.io/convolution-visualizer/

        # Skip Connection
        if in_channels == out_channels:
            self.residual_layer = nn.Identity() # Identity: https://pytorch.org/docs/stable/generated/torch.nn.Identity.html
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0) # Convusional Visualizer: https://ezyang.github.io/convolution-visualizer/

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #  x: (Batch_Size, Channel, Height, Width)

        residue = x

        x = self.groupnorm_1(x)

        x = F.silu(x) # SiLU: https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html

        x = self.conv_1(x)

        x = self.groupnorm_2(x)

        x = F.silu(x) # SiLU: https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html

        x = self.conv_2(x)

        return x + self.residual_layer(residue)

class VAE_Decoder(nn.Sequential):

    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0), # Convusional Visualizer: https://ezyang.github.io/convolution-visualizer/

            nn.Conv2d(4, 512, kernel_size=3, padding=1), # Convusional Visualizer: https://ezyang.github.io/convolution-visualizer/

            VAE_ResidualBlock(512, 512),

            VAE_AttentionBlock(512),

            VAE_ResidualBlock(512, 512),

            VAE_ResidualBlock(512, 512),

            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 4, Width / 4)
            # replicates each pixel 2 times in height and width This is why we need to use conv2d and resdidual blocks to make the image clearer
            nn.Upsample(scale_factor=2), # Upsample: https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html

            nn.Conv2d(512, 512, kernel_size=3, padding=1), # Convusional Visualizer: https://ezyang.github.io/convolution-visualizer/

            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 2, Width / 2)
            nn.Upsample(scale_factor=2), # Upsample: https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html

            nn.Conv2d(512, 512, kernel_size=3, padding=1), # Convusional Visualizer: https://ezyang.github.io/convolution-visualizer/

            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),

            # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height, Width)
            nn.Upsample(scale_factor=2), # Upsample: https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html

            nn.Conv2d(256, 256, kernel_size=3, padding=1), # Convusional Visualizer: https://ezyang.github.io/convolution-visualizer/

            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),

            nn.GroupNorm(32, 128), # Group Normalization: https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html

            nn.SiLU(), # SiLU: https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html

            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 3, Height, Width)
            nn.Conv2d(128, 3, kernel_size=3, padding=1), # Convusional Visualizer: https://ezyang.github.io/convolution-visualizer/
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch_Size, 4, Height / 8, Width / 8)
        x /= 0.18215

        for module in self:
            x = module(x)

        # (Batch_Size, 3, Height, Width)
        return x

