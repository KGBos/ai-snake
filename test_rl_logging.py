#!/usr/bin/env python3
"""
Test script to demonstrate enhanced RL logging.
Run this to see detailed real-time reinforcement learning progress.
"""

import sys
import os
import subprocess
import time
from ai_snake.utils.logging_utils import setup_logging

setup_logging(log_to_file=False, log_to_console=True, log_level='INFO')

def run_rl_test():
    """Run a short RL test with verbose logging."""
    print("🤖 AI SNAKE - REINFORCEMENT LEARNING TEST")
    print("=" * 60)
    print("This will run the learning AI for a few episodes with detailed logging.")
    print("You'll see in real-time:")
    print("  • Every decision the AI makes")
    print("  • Rewards given for each action")
    print("  • Learning progress and statistics")
    print("  • Safety interventions when AI tries to suicide")
    print("  • Episode summaries with performance analysis")
    print("=" * 60)
    
    # Run the game with learning AI and verbose logging
    cmd = [
        "python", "scripts/main.py", "play",
        "--learning",
        "--verbose", 
        "--speed", "10",  # Slower speed to see what's happening
        "--grid", "20", "20",  # Smaller grid for faster episodes
        "--auto-advance"  # Auto-restart for continuous training
    ]
    
    print("Starting RL test...")
    print("Press Ctrl+C to stop the test")
    print("=" * 60)
    
    try:
        # Run the command and capture output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Print output in real-time
        if process.stdout:
            for line in process.stdout:
                print(line.rstrip())
            
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("Test interrupted by user")
        print("=" * 60)
    except Exception as e:
        print(f"Error running test: {e}")

if __name__ == "__main__":
    run_rl_test() 