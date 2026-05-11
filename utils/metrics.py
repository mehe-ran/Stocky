import torch


def calculate_quantile_risk(predictions: torch.Tensor, targets: torch.Tensor, quantiles: list = [0.1, 0.5, 0.9]):
    risks = {}

    for i, q in enumerate(quantiles):
        pred_q = predictions[:, :, i]
        errors = targets - pred_q

        loss_q = torch.max((q - 1) * errors, q * errors).sum()
        target_sum = torch.abs(targets).sum()

        risks[f'q_{q}_risk'] = (2 * loss_q / target_sum).item() if target_sum > 0 else 0.0

    return risks