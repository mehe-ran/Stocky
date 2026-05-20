import torch
import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import textwrap
import io
import base64
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
from data_pipeline import marketdatacollector
from models.tft import temporalfusiontransformer

# initialize fastapi
app = FastAPI(title="stocky api")

# load the model globally so it stays in memory
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = temporalfusiontransformer(
    static_input_size=1, past_input_sizes={"Close": 1, "Volume": 1, "rsi_14": 1},
    future_input_sizes={"day_of_week": 1, "month": 1}, hidden_size=64
).to(device)
try:
    model.load_state_dict(torch.load("checkpoints/best_tft_model.pt", map_location=device))
    model.eval()
except Exception as e:
    print(f"[warning] could not load model weights: {e}")


class forecastrequest(BaseModel):
    ticker: str
    purchase_price: float | None = None


def fetch_and_format_live_data(ticker: str):
    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=100)
    financials = yf.Ticker(ticker).info
    pipeline = marketdatacollector(tickers=[ticker], start_date=start_date.strftime("%Y-%m-%d"),
                                   end_date=end_date.strftime("%Y-%m-%d"))
    df = pipeline.fetch_data()
    df.index = pd.to_datetime(df.index)
    vol_min, vol_max = df['volatility_7d'].min(), df['volatility_7d'].max()
    df_scaled = pipeline.scale_features(df, ['Close', 'Volume', 'returns', 'volatility_7d', 'rsi_14'])
    recent_30, recent_30_raw = df_scaled.tail(30), df.tail(30)
    past_tensor = torch.tensor(recent_30[['Close', 'Volume', 'rsi_14']].values, dtype=torch.float32).unsqueeze(0)
    past_dates = recent_30.index.tolist()
    hist_vol = recent_30_raw['volatility_7d'].values * 100
    future_dates = [end_date + dt.timedelta(days=i) for i in range(1, 15)]
    future_tensor = torch.tensor([[d.weekday(), d.month] for d in future_dates], dtype=torch.float32).unsqueeze(0)
    return torch.zeros((1, 1), dtype=torch.float32), past_tensor, future_tensor, financials, df.tail(
        1), past_dates, future_dates, hist_vol, vol_min, vol_max


@app.post("/api/forecast")
def generate_forecast(request: forecastrequest):
    try:
        ticker = request.ticker.strip().upper()
        static, past, future, financials, recent_data, past_dates, future_dates, hist_vol, vol_min, vol_max = fetch_and_format_live_data(
            ticker)

        static, past, future = static.to(device).contiguous(), past.to(device).contiguous(), future.to(
            device).contiguous()

        with torch.no_grad():
            quantiles_output = np.sort(model(static, past, future).squeeze(0).cpu().numpy(), axis=-1)

        p10 = (quantiles_output[:, 0] * (vol_max - vol_min) + vol_min) * 100
        p50 = (quantiles_output[:, 1] * (vol_max - vol_min) + vol_min) * 100
        p90 = (quantiles_output[:, 2] * (vol_max - vol_min) + vol_min) * 100

        current_price = financials.get('currentPrice', recent_data['Close'].iloc[-1])
        target_price = financials.get('targetMeanPrice', current_price)
        pe_ratio = financials.get('trailingPE', 'N/A')
        current_rsi = recent_data['rsi_14'].iloc[-1]

        if request.purchase_price:
            pnl_pct = ((current_price - request.purchase_price) / request.purchase_price) * 100
            pnl_str, avg_cost_str, position_context = f"{pnl_pct:+.2f}%", f"${request.purchase_price:.2f}", f"POSITION {pnl_str}. "
        else:
            pnl_str, avg_cost_str, position_context = "N/A", "N/A", ""

        signal, color = "HOLD", "#f39c12"
        reasoning = f"{position_context}NEUTRAL MOMENTUM. TRADING NEAR CONSENSUS FAIR VALUE."
        if target_price > (current_price * 1.05) and current_rsi < 60:
            signal, color = "BUY", "#00e676"
            reasoning = f"{position_context}DISCOUNT TO CONSENSUS > 5%. RSI INDICATES HEALTHY MOMENTUM."
        elif current_rsi >= 70:
            signal, color = "SELL (IF HELD)", "#ff1744"
            reasoning = f"{position_context}RSI OVERBOUGHT ({current_rsi:.1f} >= 70). HIGH PROBABILITY OF CONSOLIDATION."
        elif current_price > target_price:
            signal, color = "SELL (IF HELD)", "#ff1744"
            reasoning = f"{position_context}PRICE EXCEEDS CONSENSUS TARGET. MATHEMATICAL UPSIDE CAPPED."

        plt.style.use('dark_background')
        plt.rcParams.update({'font.family': 'monospace'})
        fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
        fig.patch.set_facecolor('#000000')
        ax.set_facecolor('#000000')

        ax.plot(past_dates, hist_vol, color="#666666", linewidth=2, label="HISTORICAL VOLATILITY (30D)", zorder=2)
        ax.plot([past_dates[-1], future_dates[0]], [hist_vol[-1], p50[0]], color="#00e5ff", linewidth=2, linestyle="--",
                zorder=3)
        ax.plot(future_dates, p50, label="P50 MEDIAN FORECAST", color="#00e5ff", linewidth=3, zorder=4)
        ax.fill_between(future_dates, p10, p90, color="#00e5ff", alpha=0.1, label="80% CONFIDENCE INTERVAL", zorder=1)
        ax.axhline(y=p50[0], color="#333333", linestyle="--", linewidth=1.5, zorder=0, label="T=0 BASELINE")

        ax.set_title(f"[{ticker}] VOLATILITY FORECAST & TERMINAL DATA", fontsize=16, pad=20, loc='left',
                     color="#ffffff")
        ax.set_xlabel("MARKET TIMELINE", fontsize=10, color="#666666")
        ax.set_ylabel("PROJECTED VOLATILITY (%)", fontsize=10, color="#666666")
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        fig.autofmt_xdate(rotation=0, ha='center')

        for spine in ['top', 'right']: ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']: ax.spines[spine].set_color('#333333')
        ax.tick_params(colors='#666666')
        ax.grid(True, color='#1a1a1a', linestyle='-', alpha=0.8)
        legend = ax.legend(loc="upper left", fontsize=9, facecolor='#000000', edgecolor='#333333', framealpha=1)
        for text in legend.get_texts(): text.set_color('#999999')

        pe_str = f"{pe_ratio:.2f}" if isinstance(pe_ratio, (int, float)) else pe_ratio
        metrics_text = f"TERMINAL METRICS\n------------------------------\nLAST PRICE : ${current_price:.2f}\nAVG COST   : {avg_cost_str}\nPROFIT/LOSS: {pnl_str}\nTARGET     : ${target_price:.2f}\nP/E RATIO  : {pe_str}\nRSI (14D)  : {current_rsi:.1f}\n52W HIGH   : ${financials.get('fiftyTwoWeekHigh', 'N/A')}\n52W LOW    : ${financials.get('fiftyTwoWeekLow', 'N/A')}"
        metrics_text = "\n".join([line.ljust(30) for line in metrics_text.split('\n')])
        ax.text(1.15, 0.95, metrics_text, transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='square,pad=1.2', facecolor='#000000', edgecolor='#333333', linewidth=1),
                color='#cccccc')

        signal_text = "\n".join([line.ljust(30) for line in f"ACTION: {signal}".split('\n')])
        ax.text(1.15, 0.45, signal_text, transform=ax.transAxes, fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle='square,pad=1.2', facecolor='#000000', edgecolor=color, linewidth=1.5),
                fontweight='bold', color=color)

        logic_text = "\n".join([line.ljust(30) for line in
                                f"SYSTEM LOGIC\n------------------------------\n{chr(10).join(textwrap.wrap(reasoning, width=30))}".split(
                                    '\n')])
        ax.text(1.15, 0.35, logic_text, transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='square,pad=1.2', facecolor='#000000', edgecolor='#333333', linewidth=1),
                color='#999999')

        plt.subplots_adjust(left=0.05, right=0.55)

        # save the plot to a memory buffer and convert to base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none',
                    pad_inches=0.3)
        plt.close(fig)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode('utf-8')

        return {"status": "success", "image": f"data:image/png;base64,{img_b64}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def serve_frontend():
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read(), status_code=200)


if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)