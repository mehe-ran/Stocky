import torch
import torch.nn as nn
from .grn import gatedresidualnetwork


class staticcovariateencoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout_rate: float = 0.1):
        super().__init__()

        # primary grn to process the raw static features
        self.static_grn = gatedresidualnetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout_rate=dropout_rate
        )

        # grns to generate specific context vectors for different parts of the tft

        # context for variable selection network
        self.vsn_context = gatedresidualnetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout_rate=dropout_rate
        )

        # context to initialize the lstm hidden state (h_0)
        self.lstm_hidden_context = gatedresidualnetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout_rate=dropout_rate
        )

        # context to initialize the lstm cell state (c_0)
        self.lstm_cell_context = gatedresidualnetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout_rate=dropout_rate
        )

        # context for multi-head attention enrichment
        self.attention_context = gatedresidualnetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout_rate=dropout_rate
        )

    def forward(self, static_vars):
        # pass raw static variables through the base network
        processed_static = self.static_grn(static_vars)

        # generate and return the four distinct context vectors
        return {
            "vsn": self.vsn_context(processed_static),
            "lstm_hidden": self.lstm_hidden_context(processed_static),
            "lstm_cell": self.lstm_cell_context(processed_static),
            "attention": self.attention_context(processed_static)
        }