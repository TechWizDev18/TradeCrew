import os
import sys

# ---------------------------------------------------------
# 1. REWRITE SRC/MAIN.PY (The Missing Function Fix)
# ---------------------------------------------------------
CODE_MAIN = r'''import sys
import os
# Ensure root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import glob
from src.feature_eng import add_technical_indicators
from src.model_train import train_model
from src.backtest import run_backtest

def get_all_tickers():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    files = glob.glob(os.path.join(root, "data", "*_raw.csv"))
    return [os.path.basename(f).replace("_raw.csv", "") for f in files]

# --- THIS IS THE FUNCTION YOU WERE MISSING ---
def run_portfolio_analysis(epochs=20):
    print(f"🚀 Starting Portfolio Analysis (Epochs={epochs})...")
    tickers = get_all_tickers()
    results = []
    
    for i, ticker in enumerate(tickers):
        print(f"[{i+1}/{len(tickers)}] {ticker}...", end=" ", flush=True)
        try:
            # 1. Feature Eng
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            raw_path = os.path.join(root, "data", f"{ticker}_raw.csv")
            
            # Check if processed exists
            proc_path = os.path.join(root, "data", f"{ticker}_processed.csv")
            if not os.path.exists(proc_path):
                add_technical_indicators(raw_path)
            
            # 2. Train
            train_model(ticker, epochs=int(epochs))
            
            # 3. Backtest
            final_val = run_backtest(ticker)
            
            ret = ((final_val - 10000) / 10000) * 100
            results.append({"Ticker": ticker, "Return %": round(ret, 2)})
            print(f"✅ {ret:.2f}%")
            
        except Exception as e:
            print(f"❌ {e}")

    # Report
    if results:
        df = pd.DataFrame(results).sort_values("Return %", ascending=False)
        print("\n🏆 TOP 5 STOCKS 🏆")
        print(df.head(5))
        
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        df.to_csv(os.path.join(root, "portfolio_report.csv"), index=False)
    else:
        print("⚠️ No results.")

if __name__ == "__main__":
    run_portfolio_analysis()
'''

# ---------------------------------------------------------
# 2. REWRITE SRC/TOOLS.PY (The Import Fix)
# ---------------------------------------------------------
CODE_TOOLS = r'''
try:
    from crewai.tools import BaseTool
except ImportError:
    try:
        from crewai_tools import BaseTool
    except ImportError:
        from pydantic import BaseModel
        class BaseTool(BaseModel):
            name: str
            description: str
            def _run(self, *args, **kwargs): pass

import pandas as pd
import os
import sys

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import logic
try:
    from src.main import run_portfolio_analysis
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from src.main import run_portfolio_analysis

class PortfolioAnalysisTool(BaseTool):
    name: str = "Deep Learning Portfolio Analyzer"
    description: str = "Trains LSTM models, backtests them, and returns top stocks."

    def _run(self, epochs: int = 20) -> str:
        print(f"🤖 AGENT TRIGGERED ANALYSIS (Epochs={epochs})...")
        try:
            epochs = int(epochs)
        except:
            epochs = 20
            
        run_portfolio_analysis(epochs=epochs)
        
        # Read Report
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        report_path = os.path.join(root, "portfolio_report.csv")
        
        if not os.path.exists(report_path):
             return "❌ Error: No 'portfolio_report.csv' found."
        
        try:
            df = pd.read_csv(report_path)
            top = df.head(5).to_string(index=False)
            return f"✅ Analysis Done. Top Stocks:\n\n{top}"
        except Exception as e:
            return f"❌ Error reading report: {e}"
'''

# ---------------------------------------------------------
# 3. WRITE THE FILES
# ---------------------------------------------------------
print("🔧 Restoring System Files...")

with open("src/main.py", "w", encoding="utf-8") as f:
    f.write(CODE_MAIN)
print("   ✅ Restored src/main.py")

with open("src/tools.py", "w", encoding="utf-8") as f:
    f.write(CODE_TOOLS)
print("   ✅ Restored src/tools.py")

# Clear cache to ensure Python sees the new files
import shutil
if os.path.exists("src/__pycache__"):
    shutil.rmtree("src/__pycache__")
    print("   ✅ Cleared Cache")

print("\n🎉 System Restored. Run this command now:")
print("python src/crew.py")
