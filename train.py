import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import gc


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module,
                device: str = "mps"):
    model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
        # move tensors to mps device and strictly enforce contiguous memory layout
        static = batch["static"].to(device, non_blocking=True).contiguous()
        past = batch["past"].to(device, non_blocking=True).contiguous()
        known_future = batch["known_future"].to(device, non_blocking=True).contiguous()
        target = batch["target"].to(device, non_blocking=True).contiguous()

        # zero gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(static, past, known_future)

        # calculate quantile loss for volatility forecasting
        loss = criterion(outputs, target)

        # backward pass and optimization
        loss.backward()

        # gradient clipping is highly recommended for lstms in the tft
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

        # optional: clear cache every n batches if batch size is pushing hardware limits
        # if batch_idx % 50 == 0:
        #     torch.mps.empty_cache()

    # mandatory cache clearing at the end of the epoch for apple silicon
    torch.mps.empty_cache()

    # force python garbage collection to release unreferenced tensors
    gc.collect()

    return total_loss / len(dataloader)


def validate_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: str = "mps"):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            static = batch["static"].to(device).contiguous()
            past = batch["past"].to(device).contiguous()
            known_future = batch["known_future"].to(device).contiguous()
            target = batch["target"].to(device).contiguous()

            outputs = model(static, past, known_future)
            loss = criterion(outputs, target)
            total_loss += loss.item()

    # clear validation memory before the next training epoch starts
    torch.mps.empty_cache()
    gc.collect()

    return total_loss / len(dataloader)