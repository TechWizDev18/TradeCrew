import uvicorn
import json
import os
import io 
import pandas as pd 
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse 

# --- FastAPI Initialization ---
app = FastAPI(title="AI Algo-Trading Backend")

# 1. Enable CORS for the frontend 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# --- API Endpoint to Serve the Static Report ---
@app.get("/api/report")
async def get_report():
    """
    Serves the pre-computed Deep Learning portfolio analysis report (report.json).
    """
    report_file = "report.json"
    
    if not os.path.exists(report_file):
        # We assume the user has run the Python analysis once to create this file
        raise HTTPException(status_code=500, detail=f"Report file {report_file} not found. Run the DL analysis first to generate it.")
    
    # CRITICAL FIX: Load the file and return the JSON content directly,
    # assuming the report.json file already contains the {"report": [...] } wrapper.
    try:
        with open(report_file, 'r') as f:
            data = json.load(f)
        return JSONResponse(content=data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load or parse report.json: {e}")


# --- Endpoint to serve the HTML Frontend ---
@app.get("/")
async def serve_frontend():
    """Serves the main dashboard (index.html)."""
    return FileResponse("index.html", media_type="text/html")


# --- Run Block (For local testing) ---
if __name__ == "__main__":
    print("\n--- Project is configured to run via Uvicorn ---")
    print("Run: uvicorn app:app --reload --host 0.0.0.0 --port 8000")