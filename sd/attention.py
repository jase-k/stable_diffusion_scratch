import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        # This combines the Wq, Wk and Wv matrices into one matrix
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias) # Multiplying 3 because rgb
        # This one represents the Wo matrix
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, causal_mask=False):
        # x: # (Batch_Size, Seq_Len, Dim)
        # (Batch_Size, Seq_Len, Dim)
        input_shape = x.shape 
        
        # (Batch_Size, Seq_Len, Dim)
        batch_size, sequence_length, d_embed = input_shape 

        # (Batch_Size, Seq_Len, H, Dim / H)
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head) 
        
        # x: (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, 3 * Dim) -> 3 tensors of shape (Batch_Size, Seq_Len, Dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # (Batch_Size, Seq_len, Dim) -> (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
        q = q.view(interim_shape).transpose(1, 2) # Transpose: https://pytorch.org/docs/stable/generated/torch.transpose.html
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # (Batch_Size, H, Seq_Len, Seq_Len)
        weight = q @ k.transpose(-1, -2) # This is where Queries*Keys are multiplied
    
        # Making the unrelated positions to be -inf so softmax will make them to be 0
        # [Mask Explanation Here](https://youtu.be/bCz4OMemCcA?si=-f5Qxs7dvw_LV2P5)
        if causal_mask: 
            # Mask where the upper trianble (above the principal diagonal) is made up of 1 (weight gives the shape of the matrix) (4 dimentional matrix)
            # i.e. [0, 1, 1] 
            #      [0, 0, 1]
            #      [0, 0, 0]
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1) # Upper Triangular Matrix: https://pytorch.org/docs/stable/generated/torch.triu.html
            # adding mask to the weight matrix to turn all the values in the upper triangle to -inf (during softmax, they will be 0)
            weight = weight.masked_fill_(mask, -torch.inf) # Masked Fill: https://pytorch.org/docs/stable/generated/torch.masked_fill.html

        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1) # Softmax: https://pytorch.org/docs/stable/generated/torch.softmax.html
        #  @ = matrix multiplication
        # (Batch_Size, H, Seq_Len, Seq_Len) @ (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
        output = weight @ v # This is where Queries*Keys and Values are multiplied

        # (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, Seq_Len, H, Dim / H) 
        output = output.transpose(1, 2)

        output = output.reshape(input_shape)

        output = self.out_proj(output)

        # (Batch_Size, Seq_Len, Dim)
        return output
    
class CrossAttention(nn.Module):

    # d_cross represents the Keys and Values from the outside source.
    def __init__(self, n_heads: int, d_embed: int, d_cross: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # x: (latent) (Batch_Size, Seq_Len_Q, Dim_Q)
        # y: (context [clip model text embedding]) (Batch_Size, Seq_Len_KV, Dim_KV) = (Batch_Size, 77, 768)

        input_shape = x.shape
        batch_size, seq_len, d_embed = input_shape

        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        # Multiplying query by wq ?
        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)

        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1)

        output = weight @ v

        output = output.transpose(1, 2).contiguous()

        output = output.view(input_shape)

        output = self.out_proj(output)

        return output


