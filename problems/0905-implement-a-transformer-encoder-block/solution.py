import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # Pre-attention normalisation
        self.norm1 = nn.LayerNorm(d_model)

        # Multi-head attention layer
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        # Pre-MLP Normalisation
        self.norm2 = nn.LayerNorm(d_model)
        
        # Position-wise feedforward network/MLP 
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(), # We can use nn.ReLU(), but GeLU is standard for LLMs
            nn.Linear(d_ff, d_model)
        )

        # Dropout for the residual connections
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Step 1 : Apply pre-attention normalisation on x 
        norm_x1 = self.norm1(x)
        # Step 2 : Apply MHA on norm_x1 
        # nn.MultiheadAttention returns (output, weights); we take the output [0]
        attn_out, _ = self.attn(
            norm_x1, norm_x1, norm_x1, 
            key_padding_mask=None, 
            need_weights=False
        )

        # Step 3 : Apply dropout on attn_out. This is the first residual connection 
        x = x + self.dropout(attn_out)  
        
        # Step 4 : Apply pre-MLP norm on x 
        norm_x2 = self.norm2(x)

        # Step 5 : Apply MLP on norm_x2 
        mlp_out = self.mlp(norm_x2)

        # Step 6 : Apply dropout on mlp_out. This is the second residual connection
        x = x + self.dropout(mlp_out)  
        
        return x

