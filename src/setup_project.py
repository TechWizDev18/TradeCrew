import os

# 1. Define the structure
folders = ["src", "data", "models"]
files = {
    "src/__init__.py": "",
    
    "src/data_loader.py": """
import pandas as pd
import os

def load_and_split_data(source_file="all_stocks.csv"):
    if not os.path.exists(source_file):
        print(f"❌ ERROR: '{source_file}' not found in {os.getcwd()}")
        print("   -> Please move 'all_stocks.csv' to this folder!")
        return

    print(f"📂 Loading {source_file}...")
    df = pd.read_csv(source_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df.columns = [c.lower() for c in df.columns]
    df.ffill(inplace=True)

    tickers = df['name'].unique()
    print(f"✅ Found {len(tickers)} tickers. Splitting now...")
    
    os.makedirs("data", exist_ok=True)
    for ticker in tickers:
        stock_df = df[df['name'] == ticker].copy()
        stock_df.sort_values(by='date', inplace=True)
        stock_df.to_csv(f"data/{ticker}_raw.csv", index=False)
        
    print(f"🎉 Success! Created {len(tickers)} files in /data folder.")

if __name__ == "__main__":
    load_and_split_data()
""",

    "src/feature_eng.py": """
import pandas as pd
import pandas_ta as ta
import os

def add_features(file_path):
    if not os.path.exists(file_path): return
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # Indicators
    df['RSI'] = df.ta.rsi(length=14)
    df['SMA_50'] = df.ta.sma(length=50)
    df['Target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    df.dropna(inplace=True)
    out = file_path.replace("_raw.csv", "_processed.csv")
    df.to_csv(out)
    print(f"✅ Processed {out}")

if __name__ == "__main__":
    # Process all files in data/
    import glob
    files = glob.glob("data/*_raw.csv")
    if not files:
        print("❌ No data files found. Run data_loader.py first!")
    else:
        print(f"⚙️ Processing {len(files)} files...")
        for f in files:
            add_features(f)
"""
}

# 2. Create Folders
print(f"🔨 Setting up project in: {os.getcwd()}")
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"   Created /{folder}")

# 3. Create Files
for path, content in files.items():
    with open(path, "w", encoding="utf-8") as f:
        f.write(content.strip())
    print(f"   Created {path}")

# 4. Check for Dataset
if os.path.exists("all_stocks.csv"):
    print("\n✅ 'all_stocks.csv' found! You are ready.")
    print("👉 Run this command next: python src/data_loader.py")
else:
    print("\n⚠️ 'all_stocks.csv' MISSING!")
    print(f"👉 Please move your csv file to: {os.getcwd()}")
