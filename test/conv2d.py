import torch
from torch import nn
from torch.nn import functional as F

# Conv2d Docs: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
# This file represents taking a 3 dimensional tensor and applying a single convolutional layer to it
# 3 dimensional tensors would be a common image representation as it has 3 channels (0-256 Red, Green, Blue) and 2 spatial dimensions for each pixel

class OneLayerConv2D(nn.Sequential):

    def __init__(self):
        super().__init__(
            # (Batch_Size, Channel, Height, Width) -> (Batch_Size, 128, Height, Width)
            # Conv2d Docs: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html 
            # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
            nn.Conv2d(1, 128, kernel_size=3, padding=0), # Convusional Visualizer: https://ezyang.github.io/convolution-visualizer/
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch_Size, Channel, Height, Width)
        for module in self:
            x = module(x)
        return x
    
# model = OneLayerConv2D()
# x = torch.Tensor(1, 4, 4) # docs: https://pytorch.org/docs/stable/tensors.html
# print(x.shape) # torch.Size([1, 3, 4)
# print(x) 

# x = model.forward(x)
# print(x)
# print(x.shape) # torch.Size([128, 4, 4])

from PIL import Image
import numpy as np

# Define the 10x10 grid (this is just an example, adjust the values as needed)
# 0 represents black, and 1 represents white
grid = [[
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 0, 1, 1, 0, 1],
    [1, 0, 1, 1, 0, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 0, 1, 1, 0, 1],
    [1, 0, 1, 1, 0, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]]
grid_tensor = torch.tensor(grid, dtype=torch.float32)
print(grid_tensor.shape) 
model = OneLayerConv2D()
x = model.forward(grid_tensor)
# print(x)

for i in range(128):
    print(x[i].shape)
    # Convert the grid to a numpy array and upscale it to have visible pixels
    # Each 'pixel' in the grid will actually be a 10x10 block in the final image
    grid = x[i].detach().numpy()
    grid = (grid - grid.min()) / (grid.max() - grid.min())
    pixel_data = np.repeat(np.repeat(np.array(grid) * 255, 10, axis=0), 10, axis=1)
    print(pixel_data)

    # Create a new image using Pillow, setting mode to 'L' for grayscale
    image = Image.fromarray(pixel_data.astype('uint8'), 'L')

    # Save or display the image
    image.save(f'convo_image{i}.png')
    # image.show()
