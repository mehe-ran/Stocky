import torch
from torch.utils.data import Dataset
import pandas as pd


class stockydataset(Dataset):
    def __init__(self, df: pd.DataFrame, group_col: str, static_cols: list, past_cols: list, known_future_cols: list,
                 target_col: str, max_encoder_length: int = 30, max_prediction_length: int = 14):
        self.df = df
        self.group_col = group_col
        self.static_cols = static_cols
        self.past_cols = past_cols
        self.known_future_cols = known_future_cols
        self.target_col = target_col
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length

        # total window size is past + future
        self.sequence_length = max_encoder_length + max_prediction_length

        # group data by ticker to prevent cross-contamination between stocks
        self.data_groups = {name: group for name, group in self.df.groupby(self.group_col)}
        self.valid_indices = []

        # pre-calculate valid sliding window starting points for lightning fast lookups
        for name, group in self.data_groups.items():
            group_len = len(group)
            if group_len >= self.sequence_length:
                for i in range(group_len - self.sequence_length + 1):
                    self.valid_indices.append((name, i))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        name, start_idx = self.valid_indices[idx]
        group = self.data_groups[name]

        # extract the full rolling window
        window = group.iloc[start_idx: start_idx + self.sequence_length]

        # split into the encoder (past) and decoder (future) sequences
        past_window = window.iloc[:self.max_encoder_length]
        future_window = window.iloc[self.max_encoder_length:]

        # extract tensor features
        if self.static_cols:
            static = torch.tensor(past_window[self.static_cols].iloc[0].values, dtype=torch.float32)
        else:
            # bypass apple silicon 0-element crash with a dummy tensor
            static = torch.zeros((1,), dtype=torch.float32)

        past = torch.tensor(past_window[self.past_cols].values, dtype=torch.float32)
        future = torch.tensor(future_window[self.known_future_cols].values, dtype=torch.float32)
        target = torch.tensor(future_window[self.target_col].values, dtype=torch.float32)

        # return as a dictionary to match the training loop expectation
        return {
            "static": static,
            "past": past,
            "future": future,
            "target": target
        }