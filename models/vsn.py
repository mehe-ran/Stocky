import torch
import torch.nn as nn
from .grn import gatedresidualnetwork


class variableselectionnetwork(nn.Module):
    def __init__(self, input_sizes: dict, hidden_size: int, dropout_rate: float = 0.1):
        super().__init__()

        self.hidden_size = hidden_size
        self.input_sizes = input_sizes
        self.num_inputs = len(input_sizes)

        # calculate total input dimension for the flattened weight network
        total_input_size = sum(input_sizes.values())

        # grn to generate the feature selection weights across all inputs
        self.flattened_grn = gatedresidualnetwork(
            input_size=total_input_size,
            hidden_size=hidden_size,
            output_size=self.num_inputs,
            dropout_rate=dropout_rate
        )

        # individual grns to process each specific input feature independently
        self.single_feature_grns = nn.ModuleDict({
            name: gatedresidualnetwork(
                input_size=size,
                hidden_size=hidden_size,
                output_size=hidden_size,
                dropout_rate=dropout_rate
            ) for name, size in input_sizes.items()
        })

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: dict, context=None):
        # x is a dict of tensors: {feature_name: tensor}

        # 1. compute selection weights
        # flatten all inputs along the feature dimension
        flattened_inputs = torch.cat(list(x.values()), dim=-1)

        if context is not None:
            # append static context if provided to guide temporal variable selection
            flattened_inputs = torch.cat([flattened_inputs, context], dim=-1)

        # generate softmax weights for each feature: shape (batch, time, num_inputs)
        sparse_weights = self.flattened_grn(flattened_inputs)
        sparse_weights = self.softmax(sparse_weights)

        # 2. process each feature and apply its corresponding weight
        processed_features = []
        for i, (name, tensor) in enumerate(x.items()):
            # pass feature through its dedicated grn
            processed = self.single_feature_grns[name](tensor)

            # extract weight for this specific feature and expand dimensions for broadcasting
            weight = sparse_weights[..., i].unsqueeze(-1)
            processed_features.append(processed * weight)

        # 3. sum the weighted features together
        # final shape: (batch, time, hidden_size)
        return torch.stack(processed_features, dim=-1).sum(dim=-1), sparse_weights