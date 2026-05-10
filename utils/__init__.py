import torch
import os

class earlystopping:
    def __init__(self, patience=7, min_delta=0.0, save_path='checkpoints/best_model.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def __call__(self, val_loss: float, model: torch.nn.Module):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model: torch.nn.Module):
        torch.save(model.state_dict(), self.save_path)