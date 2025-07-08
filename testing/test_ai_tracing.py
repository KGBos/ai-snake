#!/usr/bin/env python3
"""
Test script to demonstrate AI tracing functionality.
Recommended: run with --config testing/config.yaml for test defaults.
Run this to see detailed AI decision-making logs.
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config_loader import load_config
from src.game_controller import GameController

def test_ai_tracing(config):
    """Test AI tracing with a short game."""
    print("Starting AI tracing test...")
    print("=" * 50)
    
    # Use config values
    speed = config['game'].get('speed', 5)
    grid = (config['game'].get('grid_width', 10), config['game'].get('grid_height', 10))
    
    # Create game controller with AI tracing enabled
    game_controller = GameController(
        speed=speed,  # Slow speed to see decisions clearly
        ai=True,
        grid=grid,  # Small grid for easier observation
        nes_mode=False,
        ai_tracing=True
    )
    
    print("AI tracing enabled. Watch the console for decision logs.")
    print("Press Ctrl+C to stop the test early.")
    print("=" * 50)
    
    try:
        # Run a short game
        game_controller.run_game_loop()
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    
    print("=" * 50)
    print("AI tracing test completed!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AI Tracing Test")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to YAML config file')
    args = parser.parse_args()
    config = load_config(args.config)
    test_ai_tracing(config) 