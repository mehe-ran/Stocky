import torch
import torch.nn as nn


class quantileloss(nn.Module):
    def __init__(self, quantiles: list = [0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        # predictions shape: (batch_size, prediction_horizon, num_quantiles)
        # targets shape: (batch_size, prediction_horizon)

        losses = []
        for i, q in enumerate(self.quantiles):
            # extract the prediction for the specific quantile
            pred_q = predictions[:, :, i]

            # calculate the error
            errors = targets - pred_q

            # apply the pinball loss formula: max(q * error, (q - 1) * error)
            # this creates asymmetric penalties based on the target quantile
            loss_q = torch.max((q - 1) * errors, q * errors)
            losses.append(loss_q)

        # stack losses and average across all dimensions (batch, time, quantiles)
        total_loss = torch.stack(losses, dim=-1).mean()

        return total_loss