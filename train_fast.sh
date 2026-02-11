#!/bin/bash
echo "🚀 Starting FAST AI Training..."
echo "Press Ctrl+C to stop."

# Ensure logs directory exists
mkdir -p logs

# Run the game in headless mode with fast learning config
python3 scripts/main.py --config config/fast_learning.yaml play --learning --headless --auto-advance --log
