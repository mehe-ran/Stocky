import torch
import torch.nn as nn


class lstmseq2seq(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # lstm for the historical observed context (encoder)
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # lstm for the prediction horizon (decoder)
        self.decoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

    def forward(self, history_inputs, future_inputs, static_hidden, static_cell):
        # history_inputs: (batch, history_length, input_size)
        # future_inputs: (batch, prediction_length, input_size)
        # static_hidden, static_cell: context vectors from the static encoder

        # format hidden states for pytorch lstm architecture
        # lstms expect shape: (num_layers, batch, hidden_size)
        # using .contiguous() here is mandatory for mps backend stability
        h_0 = static_hidden.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        c_0 = static_cell.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()

        # 1. process historical data through the encoder
        # we capture the final state (h_n, c_n) to pass into the decoder
        encoder_output, (h_n, c_n) = self.encoder(history_inputs, (h_0, c_0))

        # 2. process known future data through the decoder
        # initialized with the final state of the history encoder
        decoder_output, _ = self.decoder(future_inputs, (h_n, c_n))

        # 3. stitch the temporal timeline back together
        # final shape: (batch, history_length + prediction_length, hidden_size)
        temporal_features = torch.cat([encoder_output, decoder_output], dim=1)

        return temporal_features