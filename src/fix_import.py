import os
import sys

# ---------------------------------------------------------
# CORRECTED CONTENT FOR SRC/TOOLS.PY
# ---------------------------------------------------------
CODE_TOOLS = r'''
# --- IMPORT FIX ---
# Try importing BaseTool from the main 'crewai' package first
try:
    from crewai.tools import BaseTool
except ImportError:
    try:
        from crewai_tools import BaseTool
    except ImportError:
        # Fallback: Create a minimal BaseTool class if imports fail
        from pydantic import BaseModel
        class BaseTool(BaseModel):
            name: str
            description: str
            def _run(self, *args, **kwargs): pass

import pandas as pd
import os
import sys

# Path Fix to ensure we can import src.main
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the portfolio logic properly
try:
    from src.main import run_portfolio_analysis
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from src.main import run_portfolio_analysis

class PortfolioAnalysisTool(BaseTool):
    name: str = "Deep Learning Portfolio Analyzer"
    description: str = (
        "Triggers the automated Deep Learning pipeline. "
        "It trains LSTM models on all available stocks, backtests them, "
        "and saves a report to 'portfolio_report.csv'. "
        "Returns the top 5 performing stocks from that report."
    )

    def _run(self, epochs: int = 20) -> str:
        print(f"🤖 AGENT IS RUNNING ANALYSIS (Epochs={epochs})... PLS WAIT.")
        
        try:
            epochs = int(epochs)
        except:
            epochs = 20
            
        run_portfolio_analysis(epochs=epochs)
        
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        report_path = os.path.join(root_dir, "portfolio_report.csv")
        
        if not os.path.exists(report_path):
            if os.path.exists("portfolio_report.csv"):
                report_path = "portfolio_report.csv"
            else:
                return "❌ Error: Analysis finished but no 'portfolio_report.csv' found."
        
        try:
            df = pd.read_csv(report_path)
            if df.empty:
                return "⚠️ Analysis completed but no profitable stocks found."
            
            top_stocks = df.head(5).to_string(index=False)
            return f"✅ Analysis Complete. Here are the Top 5 Stocks:\n\n{top_stocks}"
        except Exception as e:
            return f"❌ Error reading report: {str(e)}"
'''

print("🔧 Fixing Import Error in src/tools.py...")
with open("src/tools.py", "w", encoding="utf-8") as f:
    f.write(CODE_TOOLS.strip())

print("✅ Fix applied.")
print("👉 Now run: python src/crew.py")
