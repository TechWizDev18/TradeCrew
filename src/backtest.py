import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os

# --- 1. Define Model ---
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

# --- 2. Backtesting Logic ---
def run_backtest(ticker="AAPL", initial_capital=10000):
    print(f"📉 Starting Backtest for {ticker}...")
    
    data_path = f"data/{ticker}_processed.csv"
    model_path = f"models/{ticker}_lstm.pth"
    
    if not os.path.exists(data_path) or not os.path.exists(model_path):
        print("❌ Missing data or model. Train the model first!")
        return

    # Load & Clean Data
    df = pd.read_csv(data_path)
    ignore_cols = ['date', 'target', 'name', 'symbol']
    cols_to_drop = [c for c in df.columns if c.lower() in ignore_cols]
    df_features = df.drop(columns=cols_to_drop)
    df_features = df_features.apply(pd.to_numeric, errors='coerce')
    df_features.dropna(axis=1, how='all', inplace=True)
    df_features.dropna(inplace=True)
    
    X_raw = df_features.values
    X_min, X_max = X_raw.min(axis=0), X_raw.max(axis=0)
    X_scaled = (X_raw - X_min) / (X_max - X_min + 1e-8)
    
    SEQ_LEN = 60
    X_seq = []
    close_prices = df['close'].values[SEQ_LEN:] 
    
    for i in range(len(X_scaled) - SEQ_LEN):
        X_seq.append(X_scaled[i:i+SEQ_LEN])
        
    X_tensor = torch.Tensor(np.array(X_seq))

    # Load Model
    input_dim = X_tensor.shape[2]
    model = TradingLSTM(input_dim, hidden_dim=64, num_layers=2, output_dim=1)
    model.load_state_dict(torch.load(model_path))
    model.eval() 
    
    # Generate Predictions
    print("   🔮 Generating AI signals...")
    with torch.no_grad():
        predictions = model(X_tensor).numpy().flatten()
    
    # --- DIAGNOSTICS: Check what the model is thinking ---
    p_mean = np.mean(predictions)
    p_std = np.std(predictions)
    print(f"   📊 Stats: Mean={p_mean:.4f}, Std={p_std:.4f}, Min={predictions.min():.4f}, Max={predictions.max():.4f}")
    
    # --- ADAPTIVE STRATEGY ---
    # Buy if signal > Mean + (0.5 * StdDev)
    # Sell if signal < Mean - (0.5 * StdDev)
    buy_threshold = p_mean + (0.5 * p_std)
    sell_threshold = p_mean - (0.5 * p_std)
    print(f"   ⚙️ Adaptive Thresholds: Buy > {buy_threshold:.4f} | Sell < {sell_threshold:.4f}")

    cash = initial_capital
    position = 0 
    shares_held = 0
    trade_count = 0
    
    for i in range(len(predictions) - 1): 
        price = close_prices[i]
        signal = predictions[i]
        
        if signal > buy_threshold and position == 0:
            shares_held = cash / price
            cash = 0
            position = 1
            trade_count += 1
        elif signal < sell_threshold and position == 1:
            cash = shares_held * price
            shares_held = 0
            position = 0
            trade_count += 1
            
    # Final Value
    final_value = cash + (shares_held * close_prices[-1])
    return_pct = ((final_value - initial_capital) / initial_capital) * 100
    
    print(f"\n💰 RESULTS FOR {ticker}:")
    print(f"   Initial Capital: ${initial_capital:,.2f}")
    print(f"   Final Value:     ${final_value:,.2f}")
    print(f"   Total Return:    {return_pct:.2f}%")
    print(f"   Total Trades:    {trade_count}")
    
    return final_value

if __name__ == "__main__":
    run_backtest("AAPL")
