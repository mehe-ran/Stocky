import torch
import torch.nn as nn
from .grn import gatedresidualnetwork


class variableselectionnetwork(nn.Module):
    def __init__(self, input_sizes: dict, hidden_size: int, dropout_rate: float = 0.1):
        super().__init__()

        self.hidden_size = hidden_size
        self.input_sizes = input_sizes
        self.num_inputs = len(input_sizes)

        total_input_size = sum(input_sizes.values())
        grn_input_size = total_input_size + hidden_size

        self.flattened_grn = gatedresidualnetwork(
            input_size=grn_input_size,
            hidden_size=hidden_size,
            output_size=self.num_inputs,
            dropout_rate=dropout_rate
        )

        self.single_feature_grns = nn.ModuleDict({
            name: gatedresidualnetwork(
                input_size=size,
                hidden_size=hidden_size,
                output_size=hidden_size,
                dropout_rate=dropout_rate
            ) for name, size in input_sizes.items()
        })

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, context=None):
        flattened_inputs = x

        if context is not None:
            time_steps = x.size(1)
            expanded_context = context.unsqueeze(1).expand(-1, time_steps, -1)
            flattened_inputs = torch.cat([flattened_inputs, expanded_context], dim=-1)

        sparse_weights = self.flattened_grn(flattened_inputs)
        sparse_weights = self.softmax(sparse_weights)

        processed_features = []
        current_idx = 0

        for i, (name, size) in enumerate(self.input_sizes.items()):
            feature_tensor = x[..., current_idx: current_idx + size]
            current_idx += size
            processed = self.single_feature_grns[name](feature_tensor)
            weight = sparse_weights[..., i].unsqueeze(-1)
            processed_features.append(processed * weight)

        return torch.stack(processed_features, dim=-1).sum(dim=-1), sparse_weights