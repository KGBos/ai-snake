#!/usr/bin/env python3
"""Test trained AI performance vs untrained AI."""

import pygame
import time
from snake.game_controller import GameController

def test_ai_performance(model_path=None, episodes=10, speed=20):
    """Test AI performance with or without a trained model."""
    pygame.init()
    
    print(f"Testing AI performance for {episodes} episodes...")
    if model_path:
        print(f"Using trained model: {model_path}")
    else:
        print("Using untrained AI (random actions)")
    
    game_controller = GameController(
        speed=speed,
        ai=False,
        learning_ai=True,
        grid=(15, 15),
        auto_advance=True,
        model_path=model_path
    )
    
    if model_path and game_controller.learning_ai_controller:
        # Set to evaluation mode (no training)
        game_controller.learning_ai_controller.set_training_mode(False)
    
    total_score = 0
    total_food = 0
    episode_lengths = []
    
    for episode in range(episodes):
        game_controller.reset()
        initial_score = game_controller.game_state.score
        
        # Run episode
        game_controller.run_game_loop()
        
        # Calculate results
        final_score = game_controller.game_state.score
        food_eaten = final_score // 10  # Assuming 10 points per food
        episode_length = len(game_controller.game_state.get_snake_body())
        
        total_score += final_score
        total_food += food_eaten
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode+1}: Score={final_score}, Food={food_eaten}, Length={episode_length}")
    
    pygame.quit()
    
    # Calculate averages
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
    import argparse
    
    parser = argparse.ArgumentParser(description="Test AI performance")
    parser.add_argument('--model', type=str, help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=10, help='Number of test episodes')
    parser.add_argument('--speed', type=int, default=20, help='Game speed')
    
    args = parser.parse_args()
    
    print("🤖 AI Performance Test")
    print("=" * 40)
    
    # Test untrained AI
    print("\n1. Testing Untrained AI (Random Actions):")
    untrained_score, untrained_food, untrained_length = test_ai_performance(
        episodes=args.episodes, speed=args.speed
    )
    
    # Test trained AI if model provided
    if args.model:
        print("\n2. Testing Trained AI:")
        trained_score, trained_food, trained_length = test_ai_performance(
            model_path=args.model, episodes=args.episodes, speed=args.speed
        )
        
        print("\n📈 Comparison:")
        print(f"  Score improvement: {trained_score - untrained_score:+.1f}")
        print(f"  Food improvement: {trained_food - untrained_food:+.1f}")
        print(f"  Length improvement: {trained_length - untrained_length:+.1f}")
        
        if trained_score > untrained_score:
            print("✅ Trained AI performs better!")
        else:
            print("⚠️  Trained AI needs more training.") 