#!/bin/bash
echo "🚀 Starting AI Snake Web Dashboard..."

# 1. Start Flask in Background
python3 dashboard/app.py > logs/dashboard.log 2>&1 &
DASHBOARD_PID=$!
echo "🌐 Dashboard running at http://127.0.0.1:5000"

# 2. Open Browser (Mac)
open http://127.0.0.1:5000

# 3. Wait for user to stop Dashboard
echo "👋 Press Ctrl+C to stop Dashboard."
wait $DASHBOARD_PID
