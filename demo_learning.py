#!/usr/bin/env python3
"""
Demo script for the Snake DQN Learning AI.
This script demonstrates the learning AI system with a short training session.
"""

import torch
import sys
import os
import time

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from snake.game_controller import GameController


def demo_learning_ai():
    """Demo the learning AI system."""
    print("🐍 Snake DQN Learning AI Demo")
    print("=" * 40)
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create game controller with learning AI
    print("\nStarting learning AI mode...")
    print("Controls:")
    print("  L - Toggle learning AI on/off")
    print("  S - Save model")
    print("  ESC - Quit")
    print("  ENTER - Continue after game over")
    print("  R - Restart game")
    
    game_controller = GameController(
        speed=5,  # Slower speed so you can see the AI learning
        ai=False,
        learning_ai=True,
        grid=(15, 15),  # Smaller grid for faster learning
        auto_advance=True,  # Auto-advance for demo
        model_path=None  # Start fresh
    )
    
    # Run a few episodes
    episodes = 0
    max_episodes = 5
    
    while episodes < max_episodes:
        print(f"\n--- Episode {episodes + 1}/{max_episodes} ---")
        print("Starting new episode...")
        
        # Reset the game state for new episode
        game_controller.reset()
        
        if not game_controller.run_game_loop():
            break
        
        episodes += 1
        print(f"Episode {episodes} completed!")
        
        # Small delay between episodes
        time.sleep(2)
        
        # Show stats
        if game_controller.learning_ai_controller:
            stats = game_controller.learning_ai_controller.get_stats()
            if stats:
                print(f"Episode {episodes} Stats:")
                print(f"  Epsilon: {stats.get('epsilon', 0):.3f}")
                print(f"  Avg Reward: {stats.get('avg_reward', 0):.2f}")
                print(f"  Best Reward: {stats.get('best_reward', 0):.2f}")
                print(f"  Memory Size: {stats.get('memory_size', 0)}")
    
    # Save the model
    if game_controller.learning_ai_controller:
        game_controller.save_model("demo_model.pth")
        print("\n✅ Model saved as 'demo_model.pth'")
    
    print("\n🎉 Demo complete!")
    print("You can now:")
    print("1. Run 'python train_ai.py --episodes 1000' for longer training")
    print("2. Use 'python main.py --learning --model demo_model.pth' to continue training")
    print("3. Watch the AI learn and improve over time!")


def demo_pre_trained():
    """Demo with a pre-trained model if available."""
    model_path = "snake_dqn_model.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Pre-trained model '{model_path}' not found.")
        print("Run training first with: python train_ai.py --episodes 1000")
        return
    
    print("🤖 Loading pre-trained model...")
    
    game_controller = GameController(
        speed=10,
        ai=False,
        learning_ai=True,
        grid=(20, 20),
        auto_advance=False,
        model_path=model_path
    )
    
    # Set to evaluation mode (no training)
    if game_controller.learning_ai_controller:
        game_controller.learning_ai_controller.set_training_mode(False)
    
    print("🎮 Playing with pre-trained AI...")
    print("Watch how the AI has learned to play!")
    
    game_controller.run_game_loop()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Snake DQN Learning AI Demo')
    parser.add_argument('--pre-trained', action='store_true', 
                       help='Use pre-trained model if available')
    
    args = parser.parse_args()
    
    if args.pre_trained:
        demo_pre_trained()
    else:
        demo_learning_ai() 