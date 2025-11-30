import sys
import os

# --- PATH FIX ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crewai import Agent, Task, Crew, Process
from src.tools import portfolio_analysis_tool
from crewai.llm import LLM

# =================================================================
# 🔑 CONFIGURATION: Choose Your LLM Provider
# =================================================================
USE_GEMINI = True  # Set to False to use OpenAI instead

if USE_GEMINI:
    # OPTION 1: GEMINI (FREE with Google AI Studio)
    # Get your free API key from: https://aistudio.google.com/app/apikey
    MY_API_KEY = "AIzaSyCKpSx9jYhJJa4h2TZO8KA64otka7eAow0"
    os.environ["GEMINI_API_KEY"] = MY_API_KEY
    
    llm = LLM(
        model="gemini/gemini-1.5-flash",  # Fast and free
        # model="gemini/gemini-1.5-pro",  # More powerful, also free
        api_key=MY_API_KEY
    )
    
else:
    # OPTION 2: OPENAI (Paid)
    # Get your API key from: https://platform.openai.com/api-keys
    MY_API_KEY = "YOUR_OPENAI_API_KEY_HERE"
    os.environ["OPENAI_API_KEY"] = MY_API_KEY
    
    llm = LLM(
        model="gpt-4o-mini",  # Cheaper option
        # model="gpt-4o",     # More powerful but expensive
        api_key=MY_API_KEY
    )

# =================================================================
# 🤖 AGENT & TASK SETUP
# =================================================================

# Define Agent
analyst = Agent(
    role='Senior Quant Analyst',
    goal='Identify the most profitable stocks using Deep Learning.',
    backstory="""You are a veteran algorithmic trader at a top hedge fund. 
    You do not trust human intuition; you only trust data and models. 
    You have access to a proprietary Deep Learning engine for market analysis.""",
    verbose=True,
    memory=True,
    llm=llm,
    tools=[portfolio_analysis_tool] 
)

# Define Task
analysis_task = Task(
    description="""
    1. Use the 'Deep Learning Portfolio Analyzer' tool to analyze the market.
    2. The tool will execute the full backtesting pipeline across 30+ stocks and return the top 5 ranking results.
    3. Based on that list, write a recommendation for the #1 best performing stock.
    4. The final answer must be a professional investment recommendation and include the exact Return % achieved in the backtest.
    """,
    expected_output="A professional investment recommendation for the single best stock, detailing its achieved Return %.",
    agent=analyst
)

# Define Crew
trade_crew = Crew(
    agents=[analyst],
    tasks=[analysis_task],
    process=Process.sequential
)

# =================================================================
# 🚀 EXECUTION
# =================================================================

if __name__ == "__main__":
    print("🤖 Launching AI Agent...")
    try:
        result = trade_crew.kickoff()
        print("\n\n########################")
        print("## FINAL RECOMMENDATION ##")
        print("########################\n")
        print(result)
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Make sure your API key is valid and set correctly")
        print("2. Check if you have the required packages: pip install crewai litellm")
        print("3. Verify your internet connection")
        if USE_GEMINI:
            print("4. Get a free Gemini API key from: https://aistudio.google.com/app/apikey")
        else:
            print("4. Check your OpenAI account has credits: https://platform.openai.com/account/usage")
