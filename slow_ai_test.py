#!/usr/bin/env python3
"""
Slow AI performance test - runs games with larger window and slower speed for better visualization.
"""

import pygame
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Modify screen size to be 2x bigger
from snake.config import BASE_SCREEN_SIZE
import snake.config
snake.config.SCREEN_WIDTH = BASE_SCREEN_SIZE * 2
snake.config.SCREEN_HEIGHT = BASE_SCREEN_SIZE * 2

from snake.game_controller import GameController

def slow_test(model_path=None, num_games=10, speed=5):
    """Run a slow test with larger window and slower speed."""
    print(f"Running slow AI performance test ({num_games} games at speed {speed})...")
    
    results = []
    
    for game_num in range(num_games):
        print(f"Starting game {game_num + 1}/{num_games}...")
        
        # Create game controller with regular AI (not learning AI)
        game_controller = GameController(
            speed=speed,  # Much slower speed
            ai=True,  # Use regular AI
            learning_ai=False,  # Don't use learning AI
            grid=(15, 15),
            auto_advance=True
        )
        
        # Run the game
        game_controller.run_game_loop()
        
        # Get results
        final_score = game_controller.game_state.score
        final_length = len(game_controller.game_state.get_snake_body())
        food_eaten = final_score // 10  # Assuming 10 points per food
        
        results.append({
            'game': game_num + 1,
            'score': final_score,
            'length': final_length,
            'food': food_eaten
        })
        
        print(f"Game {game_num + 1}: Score={final_score}, Food={food_eaten}, Length={final_length}")
    
    pygame.quit()
    
    # Calculate averages
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
    import argparse
    
    parser = argparse.ArgumentParser(description="Slow AI Performance Test")
    parser.add_argument('--model', type=str, help='Path to trained model')
    parser.add_argument('--games', type=int, default=10, help='Number of games to run')
    parser.add_argument('--speed', type=int, default=5, help='Game speed (lower = slower)')
    
    args = parser.parse_args()
    
    print("🐍 Slow AI Performance Test")
    print("=" * 40)
    print(f"Window size: {snake.config.SCREEN_WIDTH}x{snake.config.SCREEN_HEIGHT}")
    print(f"Speed: {args.speed}")
    print(f"Games: {args.games}")
    print("=" * 40)
    
    results = slow_test(
        model_path=args.model,
        num_games=args.games,
        speed=args.speed
    ) 