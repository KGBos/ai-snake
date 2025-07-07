#!/usr/bin/env python3
"""
High-performance GPU training script for maximum speed.
Optimized for RTX 2080 Ti and similar GPUs.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
from datetime import datetime

from snake.learning_ai_controller import LearningAIController, RewardCalculator


class FastTrainingManager:
    """High-performance training manager optimized for GPU."""
    
    def __init__(self, grid_size=(15, 15), device='cuda', model_path=None):
        self.grid_size = grid_size
        self.device = device
        self.model_path = model_path
        
        # Initialize with GPU optimizations
        self.learning_ai = LearningAIController(
            grid_size=grid_size,
            device=device,
            model_path=model_path,
            training=True
        )
        
        # GPU optimizations
        if device == 'cuda':
            # Large batch size for GPU efficiency
            self.learning_ai.agent.batch_size = 256
            # Large memory buffer
            self.learning_ai.agent.memory_size = 50000
            # Faster learning
            self.learning_ai.agent.learning_rate = 0.002
            # Faster epsilon decay
            self.learning_ai.agent.epsilon_decay = 0.9995
        
        self.reward_calculator = RewardCalculator()
        
        # Training stats
        self.episode_rewards = []
        self.episode_lengths = []
        self.food_eaten_per_episode = []
        self.epsilon_history = []
        
        # Create output directory
        self.output_dir = f"fast_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def train_episode(self, max_steps=500):
        """Train for one episode with optimized speed."""
        from snake.models import GameState
        
        # Create game state
        game_state = GameState(grid_width=self.grid_size[0], grid_height=self.grid_size[1])
        game_state.reset()
        
        episode_reward = 0
        steps = 0
        food_eaten = 0
        
        while not game_state.game_over and steps < max_steps:
            # Get action from learning AI
            action = self.learning_ai.agent.get_action(game_state, training=True)
            direction = self.learning_ai.action_to_direction[action]
            
            # Store state for training
            state = self.learning_ai.agent.get_state_representation(game_state)
            
            # Apply action
            self._apply_action(game_state, direction)
            
            # Calculate reward
            reward = self.reward_calculator.calculate_reward(game_state, game_state.game_over)
            
            # Store experience
            next_state = self.learning_ai.agent.get_state_representation(game_state)
            self.learning_ai.agent.remember(state, action, reward, next_state, game_state.game_over)
            
            # Train the agent (more frequent training for GPU)
            if len(self.learning_ai.agent.memory) >= self.learning_ai.agent.batch_size:
                self.learning_ai.agent.replay()
            
            episode_reward += reward
            steps += 1
            
            # Track food eaten
            if game_state.score > food_eaten * 10:
                food_eaten = game_state.score // 10
        
        # Record episode stats
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(steps)
        self.food_eaten_per_episode.append(food_eaten)
        self.epsilon_history.append(self.learning_ai.agent.epsilon)
        
        return episode_reward, steps, food_eaten
    
    def _generate_food(self, game_state):
        """Generate food at random position."""
        while True:
            food = (
                np.random.randint(0, self.grid_size[0]),
                np.random.randint(0, self.grid_size[1])
            )
            if food not in game_state.snake:
                return food
    
    def _apply_action(self, game_state, direction):
        """Apply action to game state."""
        game_state.direction = direction
        game_state.move_snake()
        game_state.check_collision(0)  # Simplified time
        game_state.handle_growth()
    
    def train(self, episodes=2000, save_interval=200):
        """Fast training with GPU optimizations."""
        print(f"🚀 Starting FAST GPU training for {episodes} episodes...")
        print(f"Device: {self.device}")
        print(f"Grid size: {self.grid_size}")
        print(f"Batch size: {self.learning_ai.agent.batch_size}")
        print(f"Memory size: {self.learning_ai.agent.memory_size}")
        
        progress_bar = tqdm(range(episodes), desc="Fast Training")
        
        for episode in progress_bar:
            reward, steps, food_eaten = self.train_episode()
            
            # Update progress bar
            avg_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
            progress_bar.set_postfix({
                'Reward': f"{reward:.1f}",
                'Avg Reward': f"{avg_reward:.1f}",
                'Epsilon': f"{self.learning_ai.agent.epsilon:.3f}",
                'Food': food_eaten,
                'Memory': len(self.learning_ai.agent.memory)
            })
            
            # Save periodically
            if (episode + 1) % save_interval == 0:
                self.save_progress(episode + 1)
        
        # Final save
        self.save_progress(episodes)
        print(f"\n🏁 Fast training complete! Results saved to {self.output_dir}")
    
    def save_progress(self, episode):
        """Save training progress."""
        model_path = os.path.join(self.output_dir, f"fast_model_episode_{episode}.pth")
        self.learning_ai.save_model(model_path)
        
        stats = {
            'episode': episode,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'food_eaten_per_episode': self.food_eaten_per_episode,
            'epsilon_history': self.epsilon_history,
            'avg_reward_last_100': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            'best_reward': max(self.episode_rewards) if self.episode_rewards else 0,
            'avg_food_eaten_last_100': np.mean(self.food_eaten_per_episode[-100:]) if self.food_eaten_per_episode else 0
        }
        
        stats_path = os.path.join(self.output_dir, f"fast_stats_episode_{episode}.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)


def main():
    """Main fast training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fast GPU Training for Snake DQN')
    parser.add_argument('--episodes', type=int, default=2000, help='Number of training episodes')
    parser.add_argument('--grid-size', type=int, nargs=2, default=(15, 15), help='Grid size (width height)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--model', type=str, help='Path to pre-trained model')
    parser.add_argument('--save-interval', type=int, default=200, help='Save interval')
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Create fast training manager
    trainer = FastTrainingManager(
        grid_size=tuple(args.grid_size),
        device=args.device,
        model_path=args.model
    )
    
    # Start fast training
    trainer.train(episodes=args.episodes, save_interval=args.save_interval)


if __name__ == '__main__':
    main() 