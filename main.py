import torch
import torch.nn as nn
import argparse
from models.tft import temporalfusiontransformer
from train import train_epoch


def setup_device():
    # strict enforcement of mps backend for apple silicon
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Hardware configured: Apple Silicon (MPS) accelerated.")
    else:
        device = torch.device("cpu")
        print("Warning: MPS not available. Falling back to CPU. Performance will degrade.")
    return device


def main():
    parser = argparse.ArgumentParser(description="Stocky: Temporal Fusion Transformer")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    args = parser.parse_args()

    device = setup_device()

    # model configuration
    model = temporalfusiontransformer(
        static_input_size=5,
        past_input_sizes={"close": 1, "volume": 1, "rsi": 1},
        future_input_sizes={"day_of_week": 1, "month": 1},
        hidden_size=64,
        num_lstm_layers=2,
        num_attention_heads=4,
        num_quantiles=3  # predicting 10th, 50th, 90th percentiles
    ).to(device)

    # placeholder for actual dataloader initialization
    print("Model initialized. Ready for data pipeline injection.")

    # optimizer and loss function setup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # note: quantile loss function implementation goes here


if __name__ == "__main__":
    main()