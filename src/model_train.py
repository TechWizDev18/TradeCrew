import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os

# --- 1. The LSTM Brain ---
class TradingLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(TradingLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        
        # Fully Connected Layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # Sigmoid Activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :]) 
        return self.sigmoid(out)

# --- 2. Training Logic ---
def train_model(ticker="AAPL", epochs=100):  # <--- THIS IS THE FIX
    """
    Trains an LSTM model for a specific ticker.
    Args:
        ticker (str): The stock symbol.
        epochs (int): Number of training iterations.
    """
    file_path = f"data/{ticker}_processed.csv"
    if not os.path.exists(file_path):
        # Silently fail if file doesn't exist to keep main loop clean
        return None

    print(f"🧠 Training LSTM on {ticker} ({epochs} epochs)...")
    df = pd.read_csv(file_path)
    
    # --- NUCLEAR DATA CLEANING (Prevents crashes) ---
    ignore_cols = ['date', 'target', 'name', 'symbol']
    cols_to_drop = [c for c in df.columns if c.lower() in ignore_cols]
    df_features = df.drop(columns=cols_to_drop)

    # Force numeric
    df_features = df_features.apply(pd.to_numeric, errors='coerce')
    df_features.dropna(axis=1, how='all', inplace=True)
    df_features.dropna(inplace=True)

    # Re-align Target with cleaned features
    y_raw = df.loc[df_features.index, 'Target'].values
    X_raw = df_features.values

    # Scaling
    X_min = X_raw.min(axis=0)
    X_max = X_raw.max(axis=0)
    X_scaled = (X_raw - X_min) / (X_max - X_min + 1e-8)
    
    # Create Sequences
    SEQ_LEN = 60
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - SEQ_LEN):
        X_seq.append(X_scaled[i:i+SEQ_LEN])
        y_seq.append(y_raw[i+SEQ_LEN])
    
    if len(X_seq) == 0:
        print(f"⚠️ Not enough data for {ticker}")
        return None

    # Tensors
    X_tensor = torch.Tensor(np.array(X_seq))
    y_tensor = torch.Tensor(np.array(y_seq)).unsqueeze(1)
    
    # Model Setup
    input_dim = X_tensor.shape[2]
    model = TradingLSTM(input_dim=input_dim, hidden_dim=64, num_layers=2, output_dim=1)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training Loop
    # We suppress the per-epoch print to keep the main console clean
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

    # Save
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{ticker}_lstm.pth"
    torch.save(model.state_dict(), model_path)
    # print(f"💾 Model saved: {model_path}") 
    return model_path

if __name__ == "__main__":
    train_model("AAPL", epochs=10)
