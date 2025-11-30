import sys
import os

# --- PATH FIX ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# YOUR GEMINI KEY HERE
MY_GEMINI_KEY = "AIzaSyCKpSx9jYhJJa4h2TZO8KA64otka7eAow0"

# CRITICAL: Remove ALL OpenAI keys
for key in list(os.environ.keys()):
    if 'OPENAI' in key:
        del os.environ[key]

# Set Gemini ONLY
os.environ["GEMINI_API_KEY"] = MY_GEMINI_KEY

# Configure LiteLLM globally
import litellm
litellm.api_key = MY_GEMINI_KEY
os.environ["LITELLM_LOG"] = "ERROR"  # Reduce noise

# Import CrewAI
from crewai import Agent, Task, Crew, Process
from src.tools import portfolio_analysis_tool

class GeminiLLM:
    """Custom LLM wrapper that forces LiteLLM to use Gemini"""
    
    def __init__(self):
        self.model = "gemini/gemini-2.0-flash"
        self.temperature = 0.7
    
    def call(self, messages, **kwargs):
        """Route all calls through LiteLLM to Gemini"""
        try:
            response = litellm.completion(
                model=self.model,
                messages=messages if isinstance(messages, list) else [{"role": "user", "content": str(messages)}],
                temperature=self.temperature,
                api_key=MY_GEMINI_KEY
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ LLM Error: {e}")
            raise
    
    def __call__(self, prompt, **kwargs):
        """Support direct calling"""
        return self.call([{"role": "user", "content": prompt}], **kwargs)

# Create LLM instance
llm = GeminiLLM()

print("✅ Using: gemini-1.5-flash (via LiteLLM)")
print("✅ OpenAI: Disabled")

# =================================================================
# 🤖 AGENT & TASK
# =================================================================

analyst = Agent(
    role='Senior Quant Analyst',
    goal='Identify the most profitable stocks using Deep Learning.',
    backstory="""You are a veteran algorithmic trader at a top hedge fund. 
    You do not trust human intuition; you only trust data and models. 
    You have access to a proprietary Deep Learning engine for market analysis.""",
    verbose=True,
    memory=False,
    llm=llm,
    tools=[portfolio_analysis_tool],
    allow_delegation=False
)

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

trade_crew = Crew(
    agents=[analyst],
    tasks=[analysis_task],
    process=Process.sequential,
    verbose=True
)

# =================================================================
# 🚀 EXECUTION
# =================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 AI TRADING AGENT - POWERED BY GOOGLE GEMINI")
    print("="*70)
    
    if MY_GEMINI_KEY == "YOUR_GEMINI_API_KEY_HERE" or not MY_GEMINI_KEY:
        print("\n❌ ERROR: Set your Gemini API key first!")
        print("\n📋 Steps:")
        print("1. Get FREE key: https://aistudio.google.com/app/apikey")
        print("2. Replace 'YOUR_GEMINI_API_KEY_HERE' in line 12 with your key")
        print("3. Run again")
        exit(1)
    
    print("✓ API Key: Set")
    print("✓ Model: gemini/gemini-1.5-flash")
    print("="*70 + "\n")
    
    try:
        result = trade_crew.kickoff()
        
        print("\n\n" + "="*70)
        print("📊 FINAL RECOMMENDATION")
        print("="*70)
        print(result)
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        print("\nThis version should work. If it still fails, CrewAI is fundamentally broken.")
        print("Consider using a different framework like LangGraph or AutoGen.")
        raise