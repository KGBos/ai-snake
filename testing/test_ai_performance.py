#!/usr/bin/env python3
"""
Test trained AI performance vs untrained AI.
Recommended: run with --config testing/config.yaml for test defaults.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pygame
import time
import argparse
from src.config_loader import load_config
from src.game_controller import GameController

def test_ai_performance(config, model_path=None, episodes=None, speed=None):
    """Test AI performance with or without a trained model."""
    pygame.init()
    episodes = episodes if episodes is not None else config['testing'].get('episodes', 10)
    speed = speed if speed is not None else config['testing'].get('speed', 20)
    grid = tuple(config['testing'].get('grid_size', [15, 15]))
    print(f"Testing AI performance for {episodes} episodes...")
    if model_path:
        print(f"Using trained model: {model_path}")
    else:
        print("Using untrained AI (random actions)")
    game_controller = GameController(
        speed=speed,
        ai=False,
        learning_ai=True,
        grid=grid,
        auto_advance=True,
        model_path=model_path
    )
    if model_path and game_controller.learning_ai_controller:
        game_controller.learning_ai_controller.set_training_mode(False)
    total_score = 0
    total_food = 0
    episode_lengths = []
    for episode in range(episodes):
        game_controller.reset()
        initial_score = game_controller.game_state.score
        game_controller.run_game_loop()
        final_score = game_controller.game_state.score
        food_eaten = final_score // 10
        episode_length = len(game_controller.game_state.get_snake_body())
        total_score += final_score
        total_food += food_eaten
        episode_lengths.append(episode_length)
        print(f"Episode {episode+1}: Score={final_score}, Food={food_eaten}, Length={episode_length}")
    pygame.quit()
    avg_score = total_score / episodes
    avg_food = total_food / episodes
    avg_length = sum(episode_lengths) / len(episode_lengths)
    print(f"\n📊 Performance Summary:")
    print(f"  Average Score: {avg_score:.1f}")
    print(f"  Average Food Eaten: {avg_food:.1f}")
    print(f"  Average Snake Length: {avg_length:.1f}")
    print(f"  Best Score: {max([s for s in [game_controller.game_state.score] * episodes])}")
    return avg_score, avg_food, avg_length

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test AI performance")
    parser.add_argument('--model', type=str, help='Path to trained model')
    parser.add_argument('--episodes', type=int, help='Number of test episodes (overrides config)')
    parser.add_argument('--speed', type=int, help='Game speed (overrides config)')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to YAML config file')
    args = parser.parse_args()
    config = load_config(args.config)
    print("🤖 AI Performance Test")
    print("=" * 40)
    print("\n1. Testing Untrained AI (Random Actions):")
    untrained_score, untrained_food, untrained_length = test_ai_performance(
        config,
        episodes=args.episodes,
        speed=args.speed
    )
    if args.model:
        print("\n2. Testing Trained AI:")
        trained_score, trained_food, trained_length = test_ai_performance(
            config,
            model_path=args.model,
            episodes=args.episodes,
            speed=args.speed
        )
        print("\n📈 Comparison:")
        print(f"  Score improvement: {trained_score - untrained_score:+.1f}")
        print(f"  Food improvement: {trained_food - untrained_food:+.1f}")
        print(f"  Length improvement: {trained_length - untrained_length:+.1f}")
        if trained_score > untrained_score:
            print("✅ Trained AI performs better!")
        else:
            print("⚠️  Trained AI needs more training.") 