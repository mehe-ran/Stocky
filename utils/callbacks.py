import os
import torch
import numpy as np
from utils.logger import setup_logger

logger = setup_logger('earlystopping', 'logs/training.log')


class earlystopping:
    def __init__(self, patience: int = 5, delta: float = 0.0, save_path: str = 'checkpoints/best_tft_model.pt'):
        self.patience = patience
        self.delta = delta
        self.save_path = save_path
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

        # automatically create the checkpoint directory if it does not exist
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

    def __call__(self, val_loss: float, model: torch.nn.Module):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self._save_checkpoint(model)
        else:
            self.counter += 1
            logger.info(f"early stopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def _save_checkpoint(self, model: torch.nn.Module):
        # save model weights when validation loss decreases
        torch.save(model.state_dict(), self.save_path)
        logger.info(f"validation loss decreased. weights saved to {self.save_path}")