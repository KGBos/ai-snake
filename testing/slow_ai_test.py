#!/usr/bin/env python3
"""
Slow AI performance test - runs games with larger window and slower speed for better visualization.
Recommended: run with --config testing/config.yaml for test defaults.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pygame
import argparse
from src.config_loader import load_config
from src.game_controller import GameController

def slow_test(config, model_path=None):
    """Run a slow test with larger window and slower speed."""
    num_games = config['testing'].get('games', 10)
    speed = config['testing'].get('speed', 5)
    grid = tuple(config['testing'].get('grid_size', [15, 15]))
    print(f"Running slow AI performance test ({num_games} games at speed {speed})...")
    results = []
    for game_num in range(num_games):
        print(f"Starting game {game_num + 1}/{num_games}...")
        game_controller = GameController(
            speed=speed,
            ai=True,
            learning_ai=False,
            grid=grid,
            auto_advance=True
        )
        game_controller.run_game_loop()
        final_score = game_controller.game_state.score
        final_length = len(game_controller.game_state.get_snake_body())
        food_eaten = final_score // 10
        results.append({
            'game': game_num + 1,
            'score': final_score,
            'length': final_length,
            'food': food_eaten
        })
        print(f"Game {game_num + 1}: Score={final_score}, Food={food_eaten}, Length={final_length}")
    pygame.quit()
    avg_score = sum(r['score'] for r in results) / len(results)
    avg_food = sum(r['food'] for r in results) / len(results)
    avg_length = sum(r['length'] for r in results) / len(results)
    best_score = max(r['score'] for r in results)
    print(f"\n📊 Performance Summary:")
    print(f"  Average Score: {avg_score:.1f}")
    print(f"  Average Food Eaten: {avg_food:.1f}")
    print(f"  Average Snake Length: {avg_length:.1f}")
    print(f"  Best Score: {best_score}")
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Slow AI Performance Test")
    parser.add_argument('--model', type=str, help='Path to trained model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to YAML config file')
    args = parser.parse_args()
    config = load_config(args.config)
    print("🐍 Slow AI Performance Test")
    print("=" * 40)
    print(f"Speed: {config['testing'].get('speed', 5)}")
    print(f"Games: {config['testing'].get('games', 10)}")
    print("=" * 40)
    results = slow_test(config, model_path=args.model) 