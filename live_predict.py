import torch
import pandas as pd
import datetime as dt
from data_pipeline import marketdatacollector
from predict import forecaster
from models.tft import temporalfusiontransformer


class liveforecaster(forecaster):
    def __init__(self, model_path: str = "checkpoints/best_tft_model.pt"):
        # map device to apple silicon
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        # model dimensions must exactly match the architecture trained in main.py
        self.model = temporalfusiontransformer(
            static_input_size=1,
            past_input_sizes={"Close": 1, "Volume": 1, "rsi_14": 1},
            future_input_sizes={"day_of_week": 1, "month": 1},
            hidden_size=64
        ).to(self.device)

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()


def fetch_and_format_live_data(ticker: str):
    print(f"fetching live market data for {ticker}...")

    # pull 100 days to ensure enough padding for rsi and the 30-day window
    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=100)

    pipeline = marketdatacollector(tickers=[ticker], start_date=start_date.strftime("%Y-%m-%d"),
                                   end_date=end_date.strftime("%Y-%m-%d"))
    df = pipeline.fetch_data()

    # apply the exact same scaling used in training
    continuous_features = ['Close', 'Volume', 'returns', 'volatility_7d', 'rsi_14']
    df_scaled = pipeline.scale_features(df, continuous_features)

    # isolate the most recent 30 days for the sequence encoder
    recent_30 = df_scaled.tail(30)
    past_features = recent_30[['Close', 'Volume', 'rsi_14']].values
    past_tensor = torch.tensor(past_features, dtype=torch.float32).unsqueeze(0)

    # generate the next 14 days of calendar features for the decoder
    future_dates = [end_date + dt.timedelta(days=i) for i in range(1, 15)]
    future_features = [[d.weekday(), d.month] for d in future_dates]
    future_tensor = torch.tensor(future_features, dtype=torch.float32).unsqueeze(0)

    # generate the dummy tensor to satisfy mps memory allocation (batch_size=1, static_size=1)
    static_tensor = torch.zeros((1, 1), dtype=torch.float32)

    return static_tensor, past_tensor, future_tensor


if __name__ == "__main__":
    target_ticker = "AAPL"

    try:
        # initialize the engine with the weights we just trained
        engine = liveforecaster()

        # fetch and process real market data
        static, past, future = fetch_and_format_live_data(target_ticker)

        # generate the true forecast chart
        engine.predict_and_plot(target_ticker, static, past, future)

    except FileNotFoundError:
        print("error: best_tft_model.pt not found. waiting for main.py to finish training.")