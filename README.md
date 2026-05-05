# Stocky

Stocky is a machine learning project focused on market trend and volatility forecasting. The core of the project is a Temporal Fusion Transformer (TFT) built from scratch in PyTorch.

The codebase is specifically optimized for training on Apple Silicon (MPS). Training deep time-series models on unified memory architectures requires strict memory management, so the pipeline includes explicit contiguous memory enforcement and active garbage collection to prevent disk swapping and memory fragmentation.

## Architecture and Progress

The model separates inputs into static metadata, historical observations, and known future variables to prevent data leakage during training.

- [x] **Data Pipeline (`dataset.py`)**: Custom PyTorch Dataset handling time-window slicing.
- [x] **Gated Residual Networks (`models/grn.py`)**: Non-linear processing blocks with skip connections.
- [x] **Variable Selection Networks (`models/vsn.py`)**: Dynamic feature weighting to filter market noise.
- [x] **Static Covariate Encoders (`models/encoders.py`)**: Context vector generation from static metadata like sector or asset class.
- [x] **Seq2Seq LSTM (`models/seq2seq.py`)**: Encoder and decoder blocks bridging local temporal context.
- [x] **MPS Training Loop (`train.py`)**: Epoch loops strictly optimized for Mac hardware.
- [ ] **Multi-Head Attention (`models/attention.py`)**: Pending. Captures long-term historical dependencies.
- [ ] **TFT Main Module (`models/tft.py`)**: Pending. Main orchestrator for the network components.

## Project Structure

```text
Stocky/
├── data/               # Raw and processed financial time-series data
├── models/             # PyTorch network architecture
│   ├── __init__.py
│   ├── encoders.py
│   ├── grn.py
│   ├── seq2seq.py
│   └── vsn.py
├── dataset.py          # Windowed time-series data loader
├── train.py            # MPS-optimized training loop
├── main.py             # Execution entry point
├── README.md
└── .gitignore