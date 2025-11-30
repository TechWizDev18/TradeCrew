#!/usr/bin/env bash

# 1. RUN THE FULL ANALYSIS PIPELINE
# This executes main.py, which runs DL training/backtest and generates report.json
echo "--- Running Deep Learning Analysis to generate report.json ---"
python main.py

# Check if report.json was created
if [ ! -f report.json ]; then
    echo "ERROR: report.json file not found after running main.py. Deployment failed."
    exit 1
fi

echo "--- Analysis Complete. Starting FastAPI Server ---"

# 2. START THE FASTAPI SERVER
# This runs the uvicorn server, binding to the port provided by Render's environment.
# Render automatically sets the $PORT environment variable.
uvicorn app:app --host 0.0.0.0 --port $PORT