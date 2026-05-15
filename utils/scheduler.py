import torch

class lrscheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, patience: int = 3, factor: float = 0.5):
        # learning rate scheduler for mps optimization stability
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=factor,
            patience=patience,
            verbose=True
        )

    def step(self, val_loss: float):
        # decay learning rate dynamically based on validation plateaus
        self.scheduler.step(val_loss)