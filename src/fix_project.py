import os
import shutil
import sys
import subprocess

# ---------------------------------------------------------
# 1. DEFINE THE CORRECT CODE
# ---------------------------------------------------------

CODE_MODEL_TRAIN = r'''import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os

class TradingLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(TradingLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :]) 
        return self.sigmoid(out)

def train_model(ticker="AAPL", epochs=100):
    """
    Trains an LSTM model. Accepts 'epochs' to fix the error.
    """
    file_path = f"data/{ticker}_processed.csv"
    if not os.path.exists(file_path):
        return None

    # print(f"🧠 Training {ticker}...")
    df = pd.read_csv(file_path)
    
    # Clean Data
    ignore = ['date', 'target', 'name', 'symbol']
    drop_cols = [c for c in df.columns if c.lower() in ignore]
    df_feat = df.drop(columns=drop_cols).apply(pd.to_numeric, errors='coerce')
    df_feat.dropna(axis=1, how='all', inplace=True)
    df_feat.dropna(inplace=True)

    if df_feat.empty: return None

    y_raw = df.loc[df_feat.index, 'Target'].values
    X_raw = df_feat.values

    # Scale
    X_min, X_max = X_raw.min(axis=0), X_raw.max(axis=0)
    X_scaled = (X_raw - X_min) / (X_max - X_min + 1e-8)
    
    # Sequence
    SEQ_LEN = 60
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - SEQ_LEN):
        X_seq.append(X_scaled[i:i+SEQ_LEN])
        y_seq.append(y_raw[i+SEQ_LEN])
    
    if not X_seq: return None

    X_ten = torch.Tensor(np.array(X_seq))
    y_ten = torch.Tensor(np.array(y_seq)).unsqueeze(1)
    
    # Train
    model = TradingLSTM(X_ten.shape[2], 64, 2, 1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_ten), y_ten)
        loss.backward()
        optimizer.step()

    os.makedirs("models", exist_ok=True)
    path = f"models/{ticker}_lstm.pth"
    torch.save(model.state_dict(), path)
    return path

if __name__ == "__main__":
    train_model("AAPL", epochs=10)
'''

CODE_MAIN = r'''import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import glob
import time
from src.feature_eng import add_technical_indicators
from src.model_train import train_model
from src.backtest import run_backtest

def get_all_tickers():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    files = glob.glob(os.path.join(root, "data", "*_raw.csv"))
    return [os.path.basename(f).replace("_raw.csv", "") for f in files]

def run_portfolio():
    tickers = get_all_tickers()
    print(f"🚀 Analyzing {len(tickers)} stocks (Fast Mode)...")
    
    results = []
    
    for i, ticker in enumerate(tickers):
        print(f"[{i+1}/{len(tickers)}] {ticker}...", end=" ", flush=True)
        try:
            # 1. Feature Eng
            if not os.path.exists(f"data/{ticker}_processed.csv"):
                add_technical_indicators(f"data/{ticker}_raw.csv")
            
            # 2. Train (PASSING EPOCHS FIXES THE ERROR)
            train_model(ticker, epochs=20) 
            
            # 3. Backtest
            final_val = run_backtest(ticker)
            
            ret = ((final_val - 10000) / 10000) * 100
            results.append({"Ticker": ticker, "Return %": round(ret, 2)})
            print(f"✅ Return: {ret:.2f}%")
            
        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n🏆 LEADERBOARD 🏆")
    df = pd.DataFrame(results).sort_values("Return %", ascending=False)
    print(df.head(10))
    df.to_csv("final_report.csv", index=False)

if __name__ == "__main__":
    run_portfolio()
'''

# ---------------------------------------------------------
# 2. OVERWRITE FILES
# ---------------------------------------------------------
print("🔧 Applying fixes...")

# Fix model_train.py
with open("src/model_train.py", "w", encoding="utf-8") as f:
    f.write(CODE_MODEL_TRAIN)
print("   ✅ Fixed src/model_train.py")

# Fix main.py
with open("src/main.py", "w", encoding="utf-8") as f:
    f.write(CODE_MAIN)
print("   ✅ Fixed src/main.py")

# ---------------------------------------------------------
# 3. CLEAR CACHE (Crucial step!)
# ---------------------------------------------------------
cache_dir = "src/__pycache__"
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print("   ✅ Cleared old python cache")

# ---------------------------------------------------------
# 4. RUN THE PROJECT
# ---------------------------------------------------------
print("\n🚀 LAUNCHING PROJECT...\n")
subprocess.run([sys.executable, "src/main.py"])
