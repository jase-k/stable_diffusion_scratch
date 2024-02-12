import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention

# defining / embedding what step we are on
# this is the Time and Time Embedding block from the diffusion.png file in images folder
class TimeEmbedding(nn.Module):

    def __init__(self, n_embed: int):
        super().__init__()
        self.linear1 = nn.Linear(n_embed, 4 * n_embed) # Linear: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        self.linear2 = nn.Linear(4 * n_embed, 4 * n_embed) # Linear: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, 320)
        x = self.linear1(x)
        x = F.silu(x)
        x = self.linear2(x)
        # x: (1, 1280)
        return x

class SwitchSequential(nn.Sequential):

    # x is latent, context is the output of the clip model, time is the output of the time embedding
    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor: 
        for layer in self: 
            # This is applying the text embedding to the input
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x
    

class UNET_ResidualBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, n_time=1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels) # Group Normalization: https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1) # Convusional Visualizer: https://ezyang.github.io/convolution-visualizer/
        self.linear_time = nn.Linear(n_time, out_channels) # Linear: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html

        self.groupnorm_merged = nn.GroupNorm(32, out_channels) # Group Normalization: https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) # Convusional Visualizer: https://ezyang.github.io/convolution-visualizer/

        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            # Notice kernel size is 1 so the shape isn't changed, but the number of channels / features is
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, feature: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # feature: (Batch_Size, In_Channels, Height, Width)
        # time: (1, 1280)

        residue = feature

        feature = self.groupnorm_feature(feature)

        feature = F.silu(feature)

        feature = self.conv_feature(feature)

        time = F.silu(time)

        time = self.linear_time(time)

        # time is a 1D tensor, so we need to unsqueeze it to make it a 3D tensor
        # >>> x
        # tensor([1, 2, 3, 4])
        # >>> x.unsqueeze(-1).unsqueeze(-1)
        # tensor([[[1]],
        #         [[2]],
        #         [[3]],
        #         [[4]]])
        merged = feature + time.unsqueeze(-1).unsqueeze(-1) # unsqueeze: https://pytorch.org/docs/stable/tensors.html#torch.unsqueeze

        merged = self.groupnorm_merged(merged)

        merged = F.silu(merged)

        merged = self.conv_merged(merged)

        return merged + self.residual(residue)
    
# This is referred to as the Cross Attention Block as it adds the context from the clip model with the image
class UNET_AttentionBlock(nn.Module):

    def __init__(self, n_heads: int, n_embed: int, d_context=768):
        super().__init__()
        channels = n_heads * n_embed

        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6) # Group Normalization: https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0) # Convusional Visualizer: https://ezyang.github.io/convolution-visualizer/

        self.layernorm_1 = nn.LayerNorm(channels) # Layer Normalization: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
        self.attention_1 = SelfAttention(n_heads, channels, in_proj_bias=False) 

        self.layernorm_2 = nn.LayerNorm(channels) # Layer Normalization: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
        self.attention_2 = CrossAttention(n_heads, channels, in_proj_bias=False) 

        self.layernorm_3 = nn.LayerNorm(channels) # Layer Normalization: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html


        self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2) # Linear: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        self.linear_geglu_2 = nn.Linear(4 * channels, channels) # Linear: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0) # Convusional Visualizer: https://ezyang.github.io/convolution-visualizer/
    
    def forward(self, x, context):
        # x: (Batch_Size, Features, Height, Width)
        # context: (Batch_Size, Seq_Len, Dim)

        residue_long = x

        x = self.groupnorm(x)
        x = self.conv_input(x)

        n, c, h, w = x.shape

        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * Width)
        x = x.view(n, c, h * w)

        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Height * Width, Features)
        # >>> y
        # tensor([[1, 2, 3, 4],
        #         [2, 4, 6, 8]])
        # >>> y.transpose(-1,-2)
        # tensor([[1, 2],
        #         [2, 4],
        #         [3, 6],
        #         [4, 8]])
        x = x.transpose(-1, -2)

        # Normalization + Self Attention with skip connection

        residue_short = x

        x = self.layernorm_1(x)
        self.attention_1(x)
        x = x + residue_short

        residue_short = x

        # Normalization + Cross Attention with skip connection
        x = self.layernorm_2(x)

        # Cross Attention
        self.attention_2(x, context)

        x = x + residue_short

        residue_short = x

        # Normalization + Feed Forward with Geglue and skip connection
        x = self.layernorm_3(x)

        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)

        x = self.linerar_geglu_2(x)

        x += residue_short

        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Features, Height * Width)
        x = x.transpose(-1, -2)

        x = x.view(n, c, h, w)

        return self.conv_output(x) + residue_long


class Upsample(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1) # Convusional Visualizer: https://ezyang.github.io/convolution-visualizer/

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The interpolate function is the same as the nn.Upsample function used in the VAE_Decoder
        # x: (Batch_Size, Channel, Height, Width) -> (Batch_Size, Channel, Height * 2, Width * 2)
        x = F.interpolate(x, scale_factor=2, mode="nearest") # Interpolate: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        x = self.conv(x)

class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels) # Group Normalization: https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch_Size, 320, Height / 8 , Width / 8)
        x = self.groupnorm(x)
        x = F.silu(x)
        x = self.conv(x)
        # x: (Batch_Size, 4, Height / 8 , Width / 8)
        return x

class UNET(nn.Module):

    def __init__(self):
        super().__init__()

        # Encoder; You'll notice a similar pattern of decreasing the spatial dimensions of the image but increasing the number of channels (features)
        self.encoders = nn.Module([
            # (Batch_Size, 4, Height / 8 , Width / 8) -> (Batch_Size, 320, Height / 8 , Width / 8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),

            # Reducing the spatial dimensions of the image
            # (Batch_Size, 320, Height / 8 , Width / 8) -> (Batch_Size, 320, Height / 16 , Width / 16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),

            # Reducing the spatial dimensions of the image
            # (Batch_Size, 640, Height / 16 , Width / 16) -> (Batch_Size, 640, Height / 32 , Width / 32) 
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            # Provided you started with a 512x512 image, the spatial dimensions of the image would be 16x16 at this point

            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),

            # Reducing the spatial dimensions of the image
            # (Batch_Size, 1280, Height / 32 , Width / 32) -> (Batch_Size, 1280, Height / 64 , Width / 64) 
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            # Provided you started with a 512x512 image, the spatial dimensions of the image would be 8x8 at this point

            # Notice we dropped the UNET_AttentionBlock here meaning the last two layers are not getting the context from the clip model (text embedding)
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
        ])

        # This is the lowest point of the UNET see diffusion.png in images folder
        self.bottle_neck = SwitchSequential(
            UNET_ResidualBlock(1280, 1280),
            UNET_AttentionBlock(8, 160),
            UNET_ResidualBlock(1280, 1280),
        )

        # Decoder
        self.decoders = nn.ModuleList([
            # (Batch_Size, 2560, Height / 64 , Width / 64) -> (Batch_Size, 1280, Height / 64 , Width / 64)
            # The input channels are 2560 because we are concatenating the output of the encoder with the output of the bottle_neck. 
            # See the skip connection in the diffusion.png file in images folder represented by a gray line
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280)),

            # (Batch_Size, 2560, Height / 64 , Width / 64) -> (Batch_Size, 1280, Height / 32 , Width / 32)
            SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),

            # The output of the encoder step here was 640 so we need to concatenate that to make 1920 input channels
            # (Batch_Size, 2560, Height / 32 , Width / 32) -> (Batch_Size, 1280, Height / 16 , Width / 16)
            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), Upsample(1280)),

            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),

            # The output of the encoder step here was 320 so we need to concatenate that to make 1920 input channels
            # (Batch_Size, 960, Height / 16 , Width / 16) -> (Batch_Size, 640, Height / 8 , Width / 8)
            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), Upsample(640)),


            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),

            SwitchSequential(UNET_ResidualBlock(640 , 320), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_ResidualBlock(640 , 320), UNET_AttentionBlock(8, 40)),
        ])


# This is the unet
class Diffusion(nn.Module):

    def __init__(self): 
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320,4)

    # latent is the output of the encoder [block Z after the encoder on the diffusion.png file in images folder]
    # context is the output of the clip model [prompt embedding from the clip encoder on the diffusion.png file in images folder]
    # time is ? 
    # Remember that this forward pass needs to output a tensor the same size of the input from the encoder. The decoder will upscall that image to the original size
    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # latent: (Batch_Size, 4, Height / 8 , Width / 8)
        # context: (Batch_Size, Seq_Len, Dim)
        # time: (1, 320)

        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)

        # (Batch_Size, 4, Height / 8 , Width / 8) -> (Batch_Size, 320, Height / 8 , Width / 8)
        output = self.unet(latent, context, time)

        # (Batch_Size, 320, Height / 8 , Width / 8) -> (Batch_Size, 4, Height / 8 , Width / 8) <- Note this is the same shape as the input
        output = self.final(output)

        # (Batch_Size, 4, Height / 8 , Width / 8)
        return output

