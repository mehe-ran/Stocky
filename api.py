from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from predict import forecaster

# initialize fast api application for inference serving
app = FastAPI(title="Stocky TFT Engine API", version="1.0.0")

# load the forecasting engine globally to prevent reloading on each request
try:
    engine = forecaster(model_path="checkpoints/best_tft_model.pt")
except Exception:
    # fallback for testing without weights
    engine = forecaster()


class predictionrequest(BaseModel):
    ticker: str
    static_features: list[float]
    past_features: list[list[float]]
    future_features: list[list[float]]


@app.post("/predict")
async def get_volatility_forecast(req: predictionrequest):
    try:
        # map pydantic inputs to apple silicon contiguous tensors
        static_tensor = torch.tensor([req.static_features], dtype=torch.float32).contiguous()
        past_tensor = torch.tensor([req.past_features], dtype=torch.float32).contiguous()
        future_tensor = torch.tensor([req.future_features], dtype=torch.float32).contiguous()

        # execute forward pass via the prediction engine
        quantiles = engine.model(
            static_tensor.to(engine.device),
            past_tensor.to(engine.device),
            future_tensor.to(engine.device)
        )

        # format and return the p10, p50, and p90 confidence intervals
        q_cpu = quantiles.squeeze(0).tolist()
        return {
            "ticker": req.ticker,
            "forecast_horizon_days": len(q_cpu),
            "quantiles": q_cpu,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))