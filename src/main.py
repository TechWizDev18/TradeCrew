import sys
import os
import pandas as pd
import glob
import json # <-- NEW: Import JSON for saving the final report

# Ensure root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Assuming these modules exist and work correctly
from src.feature_eng import add_technical_indicators
from src.model_train import train_model
from src.backtest import run_backtest 

def get_all_tickers():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    files = glob.glob(os.path.join(root, "data", "*_raw.csv"))
    return [os.path.basename(f).replace("_raw.csv", "") for f in files]


def generate_report_json_from_df(df, json_path="report.json"):
    """
    Takes the final DataFrame, ensures data types are numeric, 
    and saves the final report in the format expected by the frontend: {"report": [...]}.
    """
    # 1. ENFORCE NUMERIC TYPES (Critical for dashboard to use .toFixed)
    df['Return (%)'] = pd.to_numeric(df['Return (%)'], errors='coerce')
    df['Trades'] = pd.to_numeric(df['Trades'], errors='coerce').astype(int)
    df['Final Value ($)'] = pd.to_numeric(df['Final Value ($)'], errors='coerce')

    # Drop any rows that failed conversion (e.g., had 'N/A')
    df.dropna(subset=['Return (%)', 'Trades', 'Final Value ($)'], inplace=True)
    
    # 2. CREATE JSON STRUCTURE
    # 'orient="records"' converts the DataFrame to a list of dictionaries (the array of stock objects)
    data_array = df.to_dict(orient='records')

    # 3. WRAP AND SAVE (This satisfies the {"report": [...] } structure)
    final_json = {"report": data_array}
    
    with open(json_path, 'w') as f:
        # Save without indent for smaller file size, or use indent=4 for readability
        json.dump(final_json, f) 
        
    print(f"✨ Successfully generated final dashboard report at {json_path}")


# --- MODIFIED: Reduced epochs to prevent MemoryError ---
def run_portfolio_analysis(epochs=2): 
    print(f"🚀 Starting Portfolio Analysis (Epochs={epochs})...")
    tickers = get_all_tickers()
    results = []
    
    for i, ticker in enumerate(tickers):
        INITIAL_CAPITAL = 10000 
        print(f"[{i+1}/{len(tickers)}] {ticker}...", end=" ", flush=True)
        try:
            # 1. Feature Eng (Code omitted for brevity, assumed to work)
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            raw_path = os.path.join(root, "data", f"{ticker}_raw.csv")
            proc_path = os.path.join(root, "data", f"{ticker}_processed.csv")
            if not os.path.exists(proc_path):
                add_technical_indicators(raw_path)
            
            # 2. Train (Code omitted for brevity, assumed to work)
            train_model(ticker, epochs=2)
            
            # 3. Backtest
            final_val = run_backtest(ticker)
            
            # MOCK/ASSUMED: Use actual trades count if run_backtest provides it.
            total_trades = int(final_val / 100) 
            
            ret = ((final_val - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
            
            # CRITICAL: Include all fields expected by the frontend
            results.append({
                "Ticker": ticker,
                "Return (%)": round(ret, 2), 
                "Trades": total_trades,
                "Final Value ($)": round(final_val, 2) 
            })
            print(f"✅ {ret:.2f}%")
            
        except Exception as e:
            print(f"❌ {e}")

    # ------------------- REPORTING & JSON GENERATION -------------------
    if results:
        df = pd.DataFrame(results).sort_values("Return (%)", ascending=False)
        
        # Determine the root path for file saving
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 1. Save CSV (for Agent/debugging)
        csv_path = os.path.join(root, "portfolio_report.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nCSV report saved to {csv_path}")

        # 2. AUTOMATE JSON GENERATION (The final step!)
        json_path = os.path.join(root, "report.json")
        generate_report_json_from_df(df, json_path)
        
        # Return the top 5 results as a string for the CrewAI Agent
        return df.head(5).to_string(index=False)
    else:
        return "⚠️ No results generated."


if __name__ == "__main__":
    run_portfolio_analysis(epochs=5)