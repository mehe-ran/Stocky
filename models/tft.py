import torch
import torch.nn as nn
from .encoders import staticcovariateencoder
from .vsn import variableselectionnetwork
from .seq2seq import lstmseq2seq
from .attention import interpretablemultiheadattention


class temporalfusiontransformer(nn.Module):
    def __init__(self,
                 static_input_size: int,
                 past_input_sizes: dict,
                 future_input_sizes: dict,
                 hidden_size: int,
                 num_lstm_layers: int = 1,
                 num_attention_heads: int = 4,
                 num_quantiles: int = 3,
                 dropout_rate: float = 0.1):
        super().__init__()

        # 1. static metadata processing
        self.static_encoder = staticcovariateencoder(static_input_size, hidden_size, dropout_rate)

        # 2. variable selection for time-series inputs
        self.past_vsn = variableselectionnetwork(past_input_sizes, hidden_size, dropout_rate)
        self.future_vsn = variableselectionnetwork(future_input_sizes, hidden_size, dropout_rate)

        # 3. local temporal processing
        self.seq2seq = lstmseq2seq(hidden_size, hidden_size, num_lstm_layers)

        # 4. global temporal processing
        self.attention = interpretablemultiheadattention(hidden_size, num_attention_heads, dropout_rate)

        # 5. final quantile output projection (e.g., p10, p50, p90)
        self.output_layer = nn.Linear(hidden_size, num_quantiles)

    def forward(self, static_data, past_data, known_future_data):
        # generate context vectors from static metadata
        contexts = self.static_encoder(static_data)

        # filter and weight temporal inputs
        past_features, _ = self.past_vsn(past_data, context=contexts["vsn"])
        future_features, _ = self.future_vsn(known_future_data, context=contexts["vsn"])

        # establish local time structures via lstm
        temporal_context = self.seq2seq(
            past_features,
            future_features,
            contexts["lstm_hidden"],
            contexts["lstm_cell"]
        )

        # find long-term dependencies via multi-head attention
        attention_output, _ = self.attention(temporal_context, contexts["attention"])

        # generate final quantile predictions
        # slice only the prediction horizon from the time dimension
        prediction_horizon = future_features.size(1)
        forecast_features = attention_output[:, -prediction_horizon:, :]

        return self.output_layer(forecast_features)