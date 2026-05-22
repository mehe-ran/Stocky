import os
import requests
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

# load environment variables
load_dotenv()


def fetch_fed_funds_rate():
    # retrieve api key from .env securely
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise ValueError("fred api key not found in .env file.")

    # fred api endpoint for the effective federal funds rate
    series_id = "FEDFUNDS"
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json"

    # execute api request
    response = requests.get(url)
    if response.status_code != 200:
        raise ConnectionError(f"failed to fetch macro data. status code: {response.status_code}")

    data = response.json()

    # parse the observations into a pandas dataframe
    observations = data.get("observations", [])
    df = pd.DataFrame(observations)

    # format dates to match yfinance and convert values to floats
    df['date'] = pd.to_datetime(df['date'])
    df['fed_funds_rate'] = pd.to_numeric(df['value'], errors='coerce')

    # isolate the required columns and drop nulls
    df = df[['date', 'fed_funds_rate']].dropna()

    return df


def fetch_vix(start_date="2020-01-01"):
    # pull the cboe volatility index using yfinance
    vix = yf.download("^VIX", start=start_date, progress=False)
    vix.reset_index(inplace=True)

    # we just need the date and the closing price for the macro anchor
    df = vix[['Date', 'Close']].copy()

    # flatten columns if yfinance returns a multiindex (happens in newer versions)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = ['date', 'vix_close']
    df['date'] = pd.to_datetime(df['date'])

    return df


def get_macro_anchors(start_date="2020-01-01"):
    # fetch both datasets
    rates_df = fetch_fed_funds_rate()
    vix_df = fetch_vix(start_date)

    # merge them on the date column.
    # we use 'left' to keep all trading days from the vix.
    macro_df = pd.merge(vix_df, rates_df, on='date', how='left')

    # forward-fill the monthly fed funds rate across the daily rows
    macro_df['fed_funds_rate'] = macro_df['fed_funds_rate'].ffill()

    # backfill any early dates before the first monthly rate pull
    macro_df['fed_funds_rate'] = macro_df['fed_funds_rate'].bfill()

    return macro_df


# quick test block
if __name__ == "__main__":
    print("fetching combined macro data...")
    macro_df = get_macro_anchors()
    print("macro data loaded successfully.")
    print(macro_df.tail())