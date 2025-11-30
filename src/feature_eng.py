import pandas as pd
import numpy as np
import os

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def add_technical_indicators(file_path: str):
    """
    Loads raw data, adds RSI, MACD, SMA using pure Pandas (no external lib errors).
    """
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None

    df = pd.read_csv(file_path)
    
    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    df.set_index('date', inplace=True)

    # --- 1. SMA (Simple Moving Average) ---
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    
    # --- 2. RSI (Relative Strength Index) ---
    df['RSI'] = calculate_rsi(df['close'], period=14)
    
    # --- 3. MACD (Moving Average Convergence Divergence) ---
    # EMA 12
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    # EMA 26
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    # MACD Line
    df['MACD'] = ema12 - ema26
    # Signal Line
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # --- 4. Target (Next Day's Movement: 1=UP, 0=DOWN) ---
    df['Target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # Drop NaNs (First 50 rows will be NaN due to SMA_50)
    df.dropna(inplace=True)
    
    # Save processed file
    output_path = file_path.replace("_raw.csv", "_processed.csv")
    df.to_csv(output_path)
    return output_path

if __name__ == "__main__":
    # Process ALL files in data folder
    import glob
    files = glob.glob("data/*_raw.csv")
    if not files:
        print("❌ No files found in data/ folder. Run data_loader.py first.")
    else:
        print(f"⚙️ Processing {len(files)} files...")
        for f in files:
            try:
                add_technical_indicators(f)
                print(f"   ✅ Processed {os.path.basename(f)}")
            except Exception as e:
                print(f"   ❌ Error processing {os.path.basename(f)}: {e}")
