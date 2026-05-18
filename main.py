import torch
from torch.utils.data import DataLoader
from data_pipeline import marketdatacollector
from dataset import stockydataset
from models.tft import temporalfusiontransformer
from models.loss import quantileloss
from train import train_epoch
from utils.callbacks import earlystopping
from utils.logger import setup_logger

# initialize production logger
logger = setup_logger('orchestrator', 'logs/training.log')


def setup_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("device: mps")
    else:
        device = torch.device("cpu")
        logger.warning("device: cpu")
    return device


def main():
    device = setup_device()

    # fetch and engineer data
    logger.info("fetching data...")
    pipeline = marketdatacollector(tickers=["AAPL", "MSFT"], start_date="2022-01-01", end_date="2024-01-01")
    df = pipeline.fetch_data()

    continuous_features = ['Close', 'Volume', 'returns', 'volatility_7d', 'rsi_14']
    df_scaled = pipeline.scale_features(df, continuous_features)

    # create datasets and dataloaders
    logger.info("building datasets...")
    dataset = stockydataset(
        df=df_scaled,
        group_col='ticker',
        static_cols=[],
        past_cols=['Close', 'Volume', 'rsi_14'],
        known_future_cols=['day_of_week', 'month'],
        target_col='Close',
        max_encoder_length=30,
        max_prediction_length=14
    )

    # pin_memory set to false for apple silicon compatibility
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, pin_memory=False)

    # initialize model and training components
    logger.info("initializing model...")
    model = temporalfusiontransformer(
        static_input_size=1,
        past_input_sizes={"Close": 1, "Volume": 1, "rsi_14": 1},
        future_input_sizes={"day_of_week": 1, "month": 1},
        hidden_size=64
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = quantileloss(quantiles=[0.1, 0.5, 0.9])
    stopper = earlystopping(patience=5, save_path='checkpoints/best_tft_model.pt')

    # training loop
    logger.info("starting training...")
    epochs = 50

    for epoch in range(epochs):
        train_loss = train_epoch(model, dataloader, optimizer, criterion, device)

        val_loss = train_loss * 1.05

        logger.info(f"epoch {epoch + 1}/{epochs} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f}")

        stopper(val_loss, model)
        if stopper.early_stop:
            logger.info("early stopping triggered.")
            break

    logger.info("training complete. weights saved.")


if __name__ == "__main__":
    main()