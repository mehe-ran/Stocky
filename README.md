# STOCKY 
### QUANTITATIVE FORECAST TERMINAL

> **DISCLAIMER:** this software is a technical demonstration built for educational and research purposes only. it does not constitute financial advice. market volatility is inherently unpredictable. do not trade real capital based on these algorithmic outputs.

stocky is a localized, full stack forecasting engine. it uses a temporal fusion transformer (tft) optimized for apple silicon (mps) to predict 14 day market volatility boundaries based on historical price action, volume, rsi, and temporal seasonality.

## installation and usage

**1. setup environment**
clone the repository and install the required dependencies. the engine will automatically allocate compute to the mps backend if available, and safely fallback to cpu on other hardware.

```bash
git clone [https://github.com/mehe-ran/stocky.git](https://github.com/mehe-ran/stocky.git)
cd stocky

# create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# install core dependencies
pip install -r requirements.txt