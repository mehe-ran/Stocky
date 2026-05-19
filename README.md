# Stocky

Stocky is a machine learning project focused on market trend and volatility forecasting. The core of the project is a Temporal Fusion Transformer (TFT) built from scratch in PyTorch.

The codebase is specifically optimized for training on Apple Silicon (MPS). Training deep time-series models on unified memory architectures requires strict memory management, so the pipeline includes explicit contiguous memory enforcement and active garbage collection to prevent disk swapping and memory fragmentation.

## Architecture and Progress

The model separates inputs into static metadata, historical observations, and known future variables to prevent data leakage during training.

- [x] **Data Pipeline (`data_pipeline.py` & `dataset.py`)**: Custom PyTorch Dataset handling time-window slicing and flat-index market data collection.
- [x] **Gated Residual Networks (`models/grn.py`)**: Non-linear processing blocks with skip connections.
- [x] **Variable Selection Networks (`models/vsn.py`)**: Dynamic feature weighting to filter market noise.
- [x] **Static Covariate Encoders (`models/encoders.py`)**: Context vector generation from static metadata like sector or asset class.
- [x] **Seq2Seq LSTM (`models/seq2seq.py`)**: Encoder and decoder blocks bridging local temporal context.
- [x] **MPS Training Loop (`train.py`)**: Epoch loops strictly optimized for Mac hardware with manual cache clearing.
- [x] **Multi-Head Attention (`models/attention.py`)**: Captures long-term historical dependencies.
- [x] **TFT Main Module (`models/tft.py`)**: Main orchestrator for the network components.
- [x] **Live Inference Engine (`live_predict.py`)**: Production script for fetching live market data and generating visual volatility forecasts.

## Project Structure

```text
Stocky/
├── checkpoints/        # saved model weights (.pt)
├── data/               # raw and processed financial time-series data
├── logs/               # training and system logs
├── models/             # pytorch network architecture
│   ├── __init__.py
│   ├── attention.py
│   ├── encoders.py
│   ├── grn.py
│   ├── loss.py         # quantile loss (pinball loss) implementation
│   ├── seq2seq.py
│   ├── tft.py
│   └── vsn.py
├── utils/              # support utilities
│   ├── __init__.py
│   ├── callbacks.py    # early stopping and checkpointing logic
│   ├── logger.py       # centralized logging orchestrator
│   └── metrics.py
├── data_pipeline.py    # live market data ingestion (yfinance)
├── dataset.py          # windowed time-series data loader
├── live_predict.py     # real-world inference and plotting
├── main.py             # execution entry point
├── predict.py          # inference base class
├── train.py            # mps-optimized training loop
├── README.md
└── .gitignore
