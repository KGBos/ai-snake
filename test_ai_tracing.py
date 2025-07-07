#!/usr/bin/env python3
"""
Test script to demonstrate AI tracing functionality.
Run this to see detailed AI decision-making logs.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from snake.game_controller import GameController
from snake.config import DEFAULT_GRID


def test_ai_tracing():
    """Test AI tracing with a short game."""
    print("Starting AI tracing test...")
    print("=" * 50)
    
    # Create game controller with AI tracing enabled
    game_controller = GameController(
        speed=5,  # Slow speed to see decisions clearly
        ai=True,
        grid=(10, 10),  # Small grid for easier observation
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
    test_ai_tracing() 