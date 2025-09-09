import torch
import torch.nn as nn

class TransformerNetwork(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers):
        super(TransformerNetwork, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, 2)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.fc_out(x.mean(dim=1))
        return x
