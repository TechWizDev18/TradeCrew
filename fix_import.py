import os
import sys

# ---------------------------------------------------------
# NEW CONTENT FOR SRC/TOOLS.PY
# ---------------------------------------------------------
CODE_TOOLS = r'''
# --- IMPORT FIX ---
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

# Path Fix
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.main import run_portfolio_analysis
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from src.main import run_portfolio_analysis

class PortfolioAnalysisTool(BaseTool):
    name: str = "Deep Learning Portfolio Analyzer"
    description: str = "Trains LSTM models, backtests them, and returns the top 5 performing stocks."

    def _run(self, epochs: int = 20) -> str:
        print(f"🤖 AGENT IS RUNNING ANALYSIS (Epochs={epochs})...")
        
        try:
            epochs = int(epochs)
        except:
            epochs = 20
            
        run_portfolio_analysis(epochs=epochs)
        
        # Locate report
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        report_path = os.path.join(root_dir, "portfolio_report.csv")
        
        if not os.path.exists(report_path):
            if os.path.exists("portfolio_report.csv"):
                report_path = "portfolio_report.csv"
            else:
                return "❌ Error: No report found."
        
        try:
            df = pd.read_csv(report_path)
            if df.empty:
                return "⚠️ No profitable stocks found."
            
            top_stocks = df.head(5).to_string(index=False)
            return f"✅ Analysis Complete. Top 5 Stocks:\n\n{top_stocks}"
        except Exception as e:
            return f"❌ Error reading report: {str(e)}"
'''

print("🔧 Fixing src/tools.py...")
os.makedirs("src", exist_ok=True)
with open("src/tools.py", "w", encoding="utf-8") as f:
    f.write(CODE_TOOLS.strip())

print("✅ Fix applied.")
