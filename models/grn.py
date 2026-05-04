import torch
import torch.nn as nn


class gatedresidualnetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.1):
        super().__init__()

        # primary dense layer followed by elu activation
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()

        # secondary dense layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # dropout layer for regularization
        self.dropout = nn.Dropout(dropout_rate)

        # gated linear unit (glu) components
        self.gate = nn.Linear(hidden_size, output_size * 2)
        self.glu = nn.GLU(dim=-1)

        # skip connection projection if input and output dimensions differ
        if input_size != output_size:
            self.skip_layer = nn.Linear(input_size, output_size)
        else:
            self.skip_layer = nn.Identity()

        # final layer normalization
        self.layer_norm = nn.LayerNorm(output_size)

    def forward(self, x):
        # process through primary dense layers
        hidden = self.fc1(x)
        hidden = self.elu(hidden)
        hidden = self.fc2(hidden)
        hidden = self.dropout(hidden)

        # apply the gating mechanism to filter information flow
        gated = self.gate(hidden)
        gated = self.glu(gated)

        # add the residual skip connection and apply layer normalization
        skip = self.skip_layer(x)
        return self.layer_norm(gated + skip)