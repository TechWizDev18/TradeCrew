from crewai.tools import tool
import pandas as pd
import os
import sys

# Ensure we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.main import run_portfolio_analysis

@tool("Deep Learning Portfolio Analyzer")
def portfolio_analysis_tool(epochs: int = 20) -> str:
    """
    Triggers the automated Deep Learning pipeline.
    It trains LSTM models on all available stocks, backtests them,
    and saves a report to 'portfolio_report.csv'.
    Returns the top 5 performing stocks from that report.
    
    Args:
        epochs: Number of training epochs for LSTM models (default: 20)
    
    Returns:
        A formatted string with the top 5 performing stocks
    """
    try:
        # Run the existing analysis engine
        print(f"🤖 AGENT IS RUNNING ANALYSIS (Epochs={epochs})... PLEASE WAIT.")
        run_portfolio_analysis(epochs=int(epochs))
        
        # Read the results
        report_path = "portfolio_report.csv"
        if not os.path.exists(report_path):
            return "❌ Error: Analysis finished but no report found."
        
        df = pd.read_csv(report_path)
        if df.empty:
            return "⚠️ Analysis completed but no profitable stocks found."
            
        # Return a text summary for the Agent
        top_stocks = df.head(5).to_string(index=False)
        return f"✅ Analysis Complete. Here are the Top 5 Stocks:\n\n{top_stocks}"
        
    except Exception as e:
        return f"❌ Error during analysis: {str(e)}"
