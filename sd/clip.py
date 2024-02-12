import torch
from torch import nn, rand
from torch.nn import functional as F
from attention import SelfAttention

# This file is responsible for the Input and Output Embedding block from the transformer.png file in images folder
# CLIP is responsible for encoding text embeddings and image embeddings similarly. I.e. the text a dog and an image of a dog will have similar embeddings
# This is encoding text prompts 

class CLIPEmbedding(nn.Module):
    def __init__(self, vocab_size: int, n_embed: int, n_tokens: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embed)

        # Adding positional context to the embeddings.. i.e. the position of the word in the sentence
        self.positional_embedding = nn.Parameter(torch.zeros(n_tokens, n_embed))
    
    def forward(self, tokens):
        x =  self.token_embedding(tokens)
        x = x + self.positional_embedding


class CLIPLayer(nn.Module):
    def __init__(self, n_heads: int, n_embed: int):
        super().__init__()
        
        self.layernorm1 = nn.LayerNorm(n_embed)
        self.attention = SelfAttention(n_heads, n_embed)
        self.layernorm2 = nn.LayerNorm(n_embed)
        self.linear1 = nn.Linear(n_embed, n_embed * 4)
        self.linear2 = nn.Linear(n_embed * 4, n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch_Size, Seq_Len, Dim)
        residue = x # saving original embedding for after the attention block (See transformer.png in images folder)

        # SELF ATTENTION Layer
        x = self.layernorm1(x)

        x = self.attention(x, causal_mask=True)

        x = x + residue # adding the original embedding to the attention block output

        # FeedForward Layer
        residue = x

        x = self.layernorm2(x)

        x = self.linear1(x)

        # QuickGELU activation function
        x = x * torch.sigmoid(1.702 + x) # Signmoid: https://pytorch.org/docs/stable/generated/torch.sigmoid.html

        x = self.linear2(x)

        x = x + residue

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch_Size, Seq_Len, Dim)
        residue = x

        x = self.attention(x)

        x = x + residue

        x = self.layer_norm(x)

        residue = x

        x = self.mlp(x)

        x = x + residue

        x = self.layer_norm(x)

        return x

class CLIP (nn.Module):

    def __init__(self) -> None:
        # super().__init__(*args, **kwargs)
        self.embeddding = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.Module([
            CLIPLayer(12, 768) for i in rand(12)
        ])

        self.layernorm= nn.LayerNorm(768) # Layer Normalization: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    
    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)

        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        state = self.embedding(tokens)

        for layer in self.layers: 
            state = layer(state)

        output = self.layernorm(state)

        return output
    
