#!/usr/bin/env python3
"""
Run the game with learning AI and show detailed analysis afterward.
This script runs the game and then provides comprehensive analysis for the AI assistant.
"""

import subprocess
import sys
import os
import time
from datetime import datetime

def run_game_with_analysis():
    """Run the game and then analyze the results."""
    print("🤖 AI SNAKE - LEARNING WITH ANALYSIS")
    print("=" * 60)
    print("This will run the learning AI and then provide detailed analysis.")
    print("The analysis will show:")
    print("  • Every move the AI made")
    print("  • All rewards and penalties")
    print("  • Safety interventions")
    print("  • Learning progress")
    print("  • Performance statistics")
    print("=" * 60)
    
    # Command to run the game
    cmd = [
        sys.executable, "scripts/main.py", "play",
        "--learning",
        "--speed", "10",  # Moderate speed
        "--grid", "20", "20",  # Medium grid size
        "--auto-advance"  # Auto-restart for continuous training
    ]
    
    print("Starting game...")
    print("Press Ctrl+C to stop and see analysis")
    print("=" * 60)
    
    try:
        # Run the game
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("Game stopped by user")
        print("=" * 60)
    except subprocess.CalledProcessError as e:
        print(f"Error running game: {e}")
        return
    except Exception as e:
        print(f"Unexpected error: {e}")
        return
    
    # After game ends, analyze the log
    print("\n" + "=" * 60)
    print("📊 ANALYZING GAME SESSION...")
    print("=" * 60)
    
    # Find the most recent log file
    logs_dir = "logs"
    if os.path.exists(logs_dir):
        log_files = [f for f in os.listdir(logs_dir) if f.startswith("game_session_")]
        if log_files:
            # Sort by modification time to get the most recent
            log_files.sort(key=lambda x: os.path.getmtime(os.path.join(logs_dir, x)), reverse=True)
            latest_log = os.path.join(logs_dir, log_files[0])
            
            print(f"📄 Found log file: {latest_log}")
            print("=" * 60)
            
            # Read and display the log file
            try:
                with open(latest_log, 'r') as f:
                    log_content = f.read()
                
                print("📄 GAME SESSION LOG:")
                print("=" * 60)
                print(log_content)
                print("=" * 60)
                
                # Simple analysis
                lines = log_content.split('\n')
                steps = [l for l in lines if 'STEP_' in l]
                food_events = [l for l in lines if '🍎 FOOD EATEN' in l]
                deaths = [l for l in lines if '💀 SNAKE DIED' in l]
                safety_interventions = [l for l in lines if 'SAFETY_INTERVENTION' in l]
                episodes = [l for l in lines if '🎯 EPISODE END SUMMARY' in l]
                
                print("\n📊 QUICK ANALYSIS:")
                print(f"   Total Steps: {len(steps)}")
                print(f"   Food Eaten: {len(food_events)}")
                print(f"   Deaths: {len(deaths)}")
                print(f"   Safety Interventions: {len(safety_interventions)}")
                print(f"   Episodes Completed: {len(episodes)}")
                
                if steps:
                    print(f"\n🎯 ACTION BREAKDOWN:")
                    actions = {}
                    for step in steps:
                        if 'Action=' in step:
                            action_part = step.split('Action=')[1].split()[0]
                            action = int(action_part)
                            actions[action] = actions.get(action, 0) + 1
                    
                    action_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
                    total_actions = sum(actions.values())
                    for action, count in sorted(actions.items()):
                        percentage = (count / total_actions * 100) if total_actions > 0 else 0
                        print(f"   {action_names.get(action, f'Action_{action}')}: {count} ({percentage:.1f}%)")
                
            except Exception as e:
                print(f"Error reading log file: {e}")
        else:
            print("❌ No log files found in logs directory")
    else:
        print("❌ Logs directory not found")

if __name__ == "__main__":
    run_game_with_analysis() 