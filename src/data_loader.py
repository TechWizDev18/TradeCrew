import pandas as pd
import os

def load_and_split_data(source_file: str = "all_stocks.csv"):
    """
    Reads the master CSV and splits it into individual ticker files in /data.
    """
    if not os.path.exists(source_file):
        print(f"Error: '{source_file}' not found. Please move it to the project root.")
        return []

    print(f"Loading {source_file}...")
    df = pd.read_csv(source_file)
    
    # 1. Standardize Dates
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 2. Rename columns to lowercase
    df.columns = [c.lower() for c in df.columns]
    # resulting columns: date, open, high, low, close, volume, name
    
    # 3. Handle clean-up (Forward fill missing data)
    df.ffill(inplace=True)

    # 4. Split by ticker
    tickers = df['name'].unique()
    print(f"Found {len(tickers)} unique tickers.")
    
    saved_files = []
    os.makedirs("data", exist_ok=True)

    for ticker in tickers:
        stock_df = df[df['name'] == ticker].copy()
        stock_df.sort_values(by='date', inplace=True)
        
        file_path = f"data/{ticker}_raw.csv"
        stock_df.to_csv(file_path, index=False)
        saved_files.append(file_path)
        
    print(f"Successfully created {len(saved_files)} files in /data folder.")
    return saved_files

if __name__ == "__main__":
    load_and_split_data()
