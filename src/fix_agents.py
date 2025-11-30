import os
import shutil
import sys

# ---------------------------------------------------------
# 1. CORRECT CODE FOR SRC/TOOLS.PY
# ---------------------------------------------------------
CODE_TOOLS = r'''from crewai_tools import BaseTool
import pandas as pd
import os
import sys

# Path Fix to ensure we can import src.main
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the portfolio logic properly
# We use a try/except block to handle running from different directories
try:
    from src.main import run_portfolio_analysis
except ImportError:
    # Fallback if running directly from src/
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
        # 1. Run the existing analysis engine
        print(f"🤖 AGENT IS RUNNING ANALYSIS (Epochs={epochs})... PLS WAIT.")
        
        # Ensure epochs is passed as int
        try:
            epochs = int(epochs)
        except:
            epochs = 20
            
        run_portfolio_analysis(epochs=epochs)
        
        # 2. Read the results
        # We look for the file in the project root (one level up if we are in src)
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        report_path = os.path.join(root_dir, "portfolio_report.csv")
        
        # Also check current dir just in case
        if not os.path.exists(report_path):
            if os.path.exists("portfolio_report.csv"):
                report_path = "portfolio_report.csv"
            else:
                return "❌ Error: Analysis finished but no 'portfolio_report.csv' found."
        
        try:
            df = pd.read_csv(report_path)
            if df.empty:
                return "⚠️ Analysis completed but no profitable stocks found."
                
            # 3. Return a text summary
            top_stocks = df.head(5).to_string(index=False)
            return f"✅ Analysis Complete. Here are the Top 5 Stocks:\n\n{top_stocks}"
        except Exception as e:
            return f"❌ Error reading report: {str(e)}"
'''

# ---------------------------------------------------------
# 2. CORRECT CODE FOR SRC/CREW.PY
# ---------------------------------------------------------
CODE_CREW = r'''import sys
import os

# --- PATH FIX ---
# This forces Python to see the 'TradeCrew' root folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI
from src.tools import portfolio_analysis_tool

# --- 1. SETUP LLM ---
if "GOOGLE_API_KEY" not in os.environ:
    # You can paste your key here if needed, or rely on env vars
    pass 

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    verbose=True,
    temperature=0.5,
    google_api_key=os.environ.get("GOOGLE_API_KEY")
)

# --- 2. DEFINE AGENT ---
analyst = Agent(
    role='Senior Quant Analyst',
    goal='Identify the most profitable stocks using Deep Learning.',
    backstory="""You are a veteran algorithmic trader. You rely strictly on data.
    You have a tool that runs a Deep Learning engine to backtest stocks.""",
    verbose=True,
    memory=True,
    llm=llm,
    tools=[portfolio_analysis_tool] 
)

# --- 3. DEFINE TASK ---
analysis_task = Task(
    description="""
    1. Use the 'Deep Learning Portfolio Analyzer' tool to analyze the market.
    2. The tool will return a list of top performing stocks.
    3. Based on that list, write a recommendation for the #1 best stock.
    4. Include the Return % in your final answer.
    """,
    expected_output="A recommendation for the best stock to buy based on the analysis.",
    agent=analyst
)

# --- 4. DEFINE CREW ---
trade_crew = Crew(
    agents=[analyst],
    tasks=[analysis_task],
    process=Process.sequential
)

if __name__ == "__main__":
    print("🤖 Launching AI Agent...")
    result = trade_crew.kickoff()
    print("\n\n########################")
    print("## FINAL RECOMMENDATION ##")
    print("########################\n")
    print(result)
'''

# ---------------------------------------------------------
# 3. APPLY FIXES
# ---------------------------------------------------------
print("🔧 Repairing Agent files...")

with open("src/tools.py", "w", encoding="utf-8") as f:
    f.write(CODE_TOOLS)
print("   ✅ Fixed src/tools.py")

with open("src/crew.py", "w", encoding="utf-8") as f:
    f.write(CODE_CREW)
print("   ✅ Fixed src/crew.py")

# Clear cache to force reload
cache_dir = "src/__pycache__"
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print("   ✅ Cleared cache")

print("\n🎉 Files repaired! Now run:")
print("python src/crew.py")
