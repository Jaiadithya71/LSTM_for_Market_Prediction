import pandas as pd
from TradingView_Data.TradingviewData.main import TradingViewData, Interval
from datetime import datetime
import os

# Initialize TradingViewData (assuming credentials are handled externally or use without login for free data)
tv = TradingViewData()

# Symbol lookup dictionary
symbol_lookup = {
    "TVC_DJI": ("DJI", "TVC"),
    "TVC_DXY": ("DXY", "TVC"),
    "ATHEX_DLY_FTSE": ("FTSE", "ATHEX"),
    "NASDAQ_DLY_NDX": ("NDX", "NASDAQ"),
    "TVC_NI225": ("NI225", "TVC"),
    "NSE_NIFTY": ("NIFTY", "NSE"),
    "SSE_DLY_000001": ("000001", "SSE"),
    "TVC_SPX": ("SPX", "TVC"),
    "EASYMARKETS_DAXEUR": ("DAXEUR", "EASYMARKETS")
}

def fetch_dataset(symbol_key, n_bars=6000):
    """
    Fetch n_bars of 15-min interval data for a given symbol and return as a DataFrame.

    Args:
        symbol_key (str): Key from symbol_lookup (e.g., 'TVC_DJI')
        n_bars (int): Number of bars to fetch (default: 6000)

    Returns:
        pandas.DataFrame or None: DataFrame with the fetched data, or None if fetch fails
    """
    try:
        # Get symbol and exchange from lookup
        symbol, exchange = symbol_lookup[symbol_key]

        # Fetch data
        df = tv.get_hist(
            symbol=symbol,
            exchange=exchange,
            interval=Interval.min_15,
            n_bars=n_bars
        )

        if df is None or df.empty:
            print(f"❌ No data fetched for {symbol} ({exchange})")
            return None

        # Reset index and rename columns
        df = df.reset_index()
        df.rename(columns={"datetime": "time"}, inplace=True)

        # Convert time to Asia/Kolkata timezone
        df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("Asia/Kolkata")

        # Drop unnecessary columns (e.g., volume, symbol)
        drop_cols = [col for col in df.columns if col.lower() in ["volume", "symbol"]]
        df_cleaned = df.drop(columns=drop_cols, errors="ignore")

        # Sort by time
        df_cleaned = df_cleaned.sort_values("time")

        print(f"✅ Fetched {len(df_cleaned)} records for {symbol_key}")
        return df_cleaned

    except Exception as e:
        print(f"⚠️ Error processing {symbol_key}: {str(e)}")
        return None

def fetch_datasets_for_indices(index_list, n_bars=6000):
    """
    Fetch datasets for a list of indices/stocks.

    Args:
        index_list (list): List of symbol keys (e.g., ['TVC_DJI', 'NSE_NIFTY'])
        n_bars (int): Number of bars to fetch for each symbol (default: 6000)

    Returns:
        dict: Dictionary mapping symbol_key to its DataFrame (or None if fetch failed)
    """
    # Validate input indices
    invalid_indices = [idx for idx in index_list if idx not in symbol_lookup]
    if invalid_indices:
        print(f"❌ Invalid indices: {invalid_indices}")
        return {}

    # Fetch datasets
    datasets = {}
    for symbol_key in index_list:
        print(f"\n📥 Processing {symbol_key}...")
        df = fetch_dataset(symbol_key, n_bars)
        datasets[symbol_key] = df

    return datasets

def save_datasets(datasets, output_dir="/content"):
    """
    Save datasets to CSV files following the naming convention.

    Args:
        datasets (dict): Dictionary of symbol_key to DataFrame
        output_dir (str): Directory to save the CSV files
    """
    os.makedirs(output_dir, exist_ok=True)
    for symbol_key, df in datasets.items():
        if df is not None and not df.empty:
            csv_filename = f"{symbol_key}_15_15min.csv"
            output_path = os.path.join(output_dir, csv_filename)
            df.to_csv(output_path, index=False)
            print(f"✅ Dataset saved: {output_path} ({len(df)} records)")

def fetch_live_data(symbol_key, n_bars=1):
    """
    Fetch the latest 15-min interval candle for a given symbol and return as a DataFrame.

    Args:
        symbol_key (str): Key from symbol_lookup (e.g., 'TVC_DJI')
        n_bars (int): Number of bars to fetch (default: 1 for the latest candle)

    Returns:
        pandas.DataFrame or None: Single-row DataFrame with the latest candle, or None if fetch fails
    """
    try:
        # Get symbol and exchange from lookup
        symbol, exchange = symbol_lookup[symbol_key]

        # Fetch data (only the latest candle)
        df = tv.get_hist(
            symbol=symbol,
            exchange=exchange,
            interval=Interval.min_15,
            n_bars=n_bars
        )

        if df is None or df.empty:
            print(f"❌ No data fetched for {symbol} ({exchange})")
            return None

        # Reset index and rename columns
        df = df.reset_index()
        df.rename(columns={"datetime": "time"}, inplace=True)

        # Convert time to Asia/Kolkata timezone
        df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("Asia/Kolkata")

        # Drop unnecessary columns (e.g., volume, symbol)
        drop_cols = [col for col in df.columns if col.lower() in ["volume", "symbol"]]
        df_cleaned = df.drop(columns=drop_cols, errors="ignore")

        # Sort by time and get the latest candle
        df_cleaned = df_cleaned.sort_values("time")
        latest_candle = df_cleaned.tail(1)  # Return the last row as a DataFrame

        print(f"✅ Fetched latest candle for {symbol_key}")
        return latest_candle

    except Exception as e:
        print(f"⚠️ Error processing {symbol_key}: {str(e)}")
        return None
