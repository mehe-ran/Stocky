import torch
import matplotlib.pyplot as plt
import numpy as np
from models.tft import temporalfusiontransformer


class forecaster:
    def __init__(self, model_path: str = None, device: str = "mps"):
        self.device = torch.device(device if torch.backends.mps.is_available() else "cpu")

        self.model = temporalfusiontransformer(
            static_input_size=5,
            past_input_sizes={"close": 1, "volume": 1, "rsi": 1},
            future_input_sizes={"day_of_week": 1, "month": 1},
            hidden_size=64,
            num_lstm_layers=2,
            num_attention_heads=4,
            num_quantiles=3
        ).to(self.device)

        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        self.model.eval()

    def predict_and_plot(self, ticker: str, static_data: torch.Tensor, past_data: torch.Tensor,
                         future_data: torch.Tensor):
        print(f"Generating volatility forecast for {ticker}...")

        with torch.no_grad():
            static = static_data.to(self.device).contiguous()
            past = past_data.to(self.device).contiguous()
            future = future_data.to(self.device).contiguous()

            quantiles = self.model(static, past, future)
            quantiles_cpu = quantiles.squeeze(0).cpu().numpy()

        self._plot_fan_chart(ticker, quantiles_cpu)

    def _plot_fan_chart(self, ticker: str, quantiles: np.ndarray):
        horizon = quantiles.shape[0]
        days = np.arange(1, horizon + 1)

        p10 = quantiles[:, 0]
        p50 = quantiles[:, 1]
        p90 = quantiles[:, 2]

        plt.figure(figsize=(10, 6))
        plt.plot(days, p50, color='#1f77b4', linewidth=2, label='Median Forecast (P50)')
        plt.fill_between(days, p10, p90, color='#1f77b4', alpha=0.2, label='80% Confidence Interval (P10-P90)')

        plt.title(f"{ticker} Volatility Forecast")
        plt.xlabel("Future Time Steps (Days)")
        plt.ylabel("Scaled Value")
        plt.legend(loc="upper left")
        plt.grid(alpha=0.3)

        output_file = f"{ticker}_forecast.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Forecast chart successfully saved to {output_file}")


if __name__ == "__main__":
    print("Prediction engine initialized.")