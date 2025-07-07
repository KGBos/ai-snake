#!/usr/bin/env python3
"""
Quick AI performance test - runs 10 games at high speed.
"""

from ai_performance_test import AIPerformanceTester

def quick_test():
    """Run a quick 10-game test."""
    print("Running quick AI performance test (10 games)...")
    
    tester = AIPerformanceTester(
        num_games=10,
        speed=100,  # Very fast
        grid_size=(15, 15)  # Smaller grid for faster games
    )
    
    # Ensure auto_advance is set for all games
    tester.run_all_games()
    tester.analyze_results()

if __name__ == '__main__':
    quick_test() 