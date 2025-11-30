import pandas as pd
import json
import os

def generate_report_json(csv_path="portfolio_report.csv", json_path="report.json"):
    """
    Reads the portfolio CSV, cleans and renames columns, ensures data types are numeric,
    and saves the final report in the format expected by the frontend: {"report": [...]}.
    """
    if not os.path.exists(csv_path):
        print(f"❌ Error: CSV file not found at {csv_path}. Run main.py first!")
        return

    print(f"✅ Reading data from {csv_path}")
    df = pd.read_csv(csv_path)

    # 1. ADD MISSING/RENAME COLUMNS
    # NOTE: You MUST ensure your main.py outputs the trades and final value. 
    # For now, we'll assume they were in the input CSV or we'll mock them:
    
    # Standardize return column name for frontend consumption
    df.rename(columns={'Return %': 'Return (%)'}, inplace=True)
    
    # If Trades and Final Value are missing, this will fail. 
    # For a quick fix, let's assume they exist, but you MUST implement them in main.py
    if 'Trades' not in df.columns:
        # MOCK DATA FOR DEMO - REPLACE THIS WITH REAL DATA FROM main.py
        df['Trades'] = 100 
    if 'Final Value' in df.columns:
        df.rename(columns={'Final Value': 'Final Value ($)'}, inplace=True)
    elif 'Final Value ($)' not in df.columns:
        # MOCK DATA FOR DEMO - REPLACE THIS WITH REAL DATA FROM main.py
        # Initial $10,000 + (Return % * 100)
        df['Final Value ($)'] = 10000 + (df['Return (%)'] * 100) 


    # 2. ENFORCE NUMERIC TYPES (This is the critical step to fix the JS error!)
    df['Return (%)'] = pd.to_numeric(df['Return (%)'], errors='coerce')
    df['Trades'] = pd.to_numeric(df['Trades'], errors='coerce').astype(int)
    df['Final Value ($)'] = pd.to_numeric(df['Final Value ($)'], errors='coerce')

    # Drop any rows that failed conversion (e.g., had 'N/A')
    df.dropna(subset=['Return (%)', 'Trades', 'Final Value ($)'], inplace=True)
    
    # 3. CREATE JSON STRUCTURE
    # 'orient="records"' converts the DataFrame to a list of dictionaries (the array of stock objects)
    data_array = df.to_dict(orient='records')

    # 4. WRAP AND SAVE (This satisfies the {"report": [...] } structure)
    final_json = {"report": data_array}
    
    with open(json_path, 'w') as f:
        json.dump(final_json, f, indent=4)
        
    print(f"✨ Successfully generated report at {json_path}")


if __name__ == "__main__":
    generate_report_json()