#!/usr/bin/env python3
"""
Quick RL test with enhanced logging.
Run this to see real-time reinforcement learning in action.
"""

import subprocess
import sys
import os

def run_quick_test():
    """Run a quick RL test with verbose logging."""
    print("🤖 QUICK RL TEST WITH ENHANCED LOGGING")
    print("=" * 60)
    print("This will run the learning AI for a few episodes.")
    print("Watch the console for detailed real-time logs!")
    print("=" * 60)
    
    # Command to run the game with learning AI and verbose logging
    cmd = [
        sys.executable, "scripts/main.py", "play",
        "--learning",
        "--verbose",
        "--speed", "8",  # Moderate speed to see what's happening
        "--grid", "15", "15",  # Small grid for faster episodes
        "--auto-advance"  # Auto-restart for continuous training
    ]
    
    print("Starting RL test...")
    print("You'll see:")
    print("  • Real-time decision logs every 10 steps")
    print("  • Reward breakdowns every 100 steps")
    print("  • Food rewards and death penalties")
    print("  • Safety interventions when AI tries to suicide")
    print("  • Episode summaries with learning progress")
    print()
    print("Press Ctrl+C to stop the test")
    print("=" * 60)
    
    try:
        # Run the command
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("Test stopped by user")
        print("=" * 60)
    except subprocess.CalledProcessError as e:
        print(f"Error running test: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    run_quick_test() 