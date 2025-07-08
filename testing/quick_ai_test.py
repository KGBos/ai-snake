#!/usr/bin/env python3
"""
Quick AI performance test - runs 10 games at high speed.
Recommended: run with --config testing/config.yaml for test defaults.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai_performance_test import AIPerformanceTester
from src.config_loader import load_config
import argparse

def quick_test(config):
    """Run a quick 10-game test."""
    print("Running quick AI performance test (10 games)...")
    tester = AIPerformanceTester(
        num_games=config['testing'].get('games', 10),
        speed=config['testing'].get('speed', 100),
        grid_size=tuple(config['testing'].get('grid_size', [15, 15]))
    )
    tester.run_all_games()
    tester.analyze_results()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Quick AI Test")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to YAML config file')
    args = parser.parse_args()
    config = load_config(args.config)
    quick_test(config) 