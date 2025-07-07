#!/usr/bin/env python3
"""
Memory-optimized training script for systems with limited RAM.
"""

import torch
import numpy as np
from tqdm import tqdm
import os
import json
from datetime import datetime
import gc  # Garbage collection

from snake.learning_ai_controller import LearningAIController, RewardCalculator


class MemoryOptimizedTrainer:
    """Memory-efficient training manager."""
    
    def __init__(self, grid_size=(15, 15), device='cuda', model_path=None):
        self.grid_size = grid_size
        self.device = device
        self.model_path = model_path
        
        # Initialize with memory optimizations
        self.learning_ai = LearningAIController(
            grid_size=grid_size,
            device=device,
            model_path=model_path,
            training=True
        )
        
        # Memory optimizations
        self.learning_ai.agent.memory_size = 5000  # Smaller buffer
        self.learning_ai.agent.batch_size = 32      # Smaller batches
        self.reward_calculator = RewardCalculator()
        
        # Minimal stats storage
        self.episode_rewards = []
        self.episode_lengths = []
        
        # Create output directory
        self.output_dir = f"memory_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def train_episode(self, max_steps=300):
        """Train for one episode with minimal memory usage."""
        from snake.models import GameState
        
        # Create game state
        game_state = GameState(grid_width=self.grid_size[0], grid_height=self.grid_size[1])
        game_state.reset()
        
        episode_reward = 0
        steps = 0
        
        while not game_state.game_over and steps < max_steps:
            # Get action
            action = self.learning_ai.agent.get_action(game_state, training=True)
            direction = self.learning_ai.action_to_direction[action]
            
            # Store state
            state = self.learning_ai.agent.get_state_representation(game_state)
            
            # Apply action
            self._apply_action(game_state, direction)
            
            # Calculate reward
            reward = self.reward_calculator.calculate_reward(game_state, game_state.game_over)
            
            # Store experience
            next_state = self.learning_ai.agent.get_state_representation(game_state)
            self.learning_ai.agent.remember(state, action, reward, next_state, game_state.game_over)
            
            # Train occasionally (not every step to save memory)
            if len(self.learning_ai.agent.memory) >= self.learning_ai.agent.batch_size and steps % 5 == 0:
                self.learning_ai.agent.replay()
            
            episode_reward += reward
            steps += 1
        
        # Force garbage collection
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        
        return episode_reward, steps
    
    def _apply_action(self, game_state, direction):
        """Apply action to game state."""
        game_state.direction = direction
        game_state.move_snake()
        game_state.check_collision(0)
        game_state.handle_growth()
    
    def train(self, episodes=1000, save_interval=100):
        """Memory-optimized training."""
        print(f"🧠 Starting memory-optimized training for {episodes} episodes...")
        print(f"Device: {self.device}")
        print(f"Memory buffer: {self.learning_ai.agent.memory_size}")
        print(f"Batch size: {self.learning_ai.agent.batch_size}")
        
        progress_bar = tqdm(range(episodes), desc="Memory-Optimized Training")
        
        for episode in progress_bar:
            reward, steps = self.train_episode()
            
            # Store minimal stats
            self.episode_rewards.append(reward)
            self.episode_lengths.append(steps)
            
            # Update progress bar
            avg_reward = np.mean(self.episode_rewards[-50:]) if self.episode_rewards else 0
            progress_bar.set_postfix({
                'Reward': f"{reward:.1f}",
                'Avg Reward': f"{avg_reward:.1f}",
                'Epsilon': f"{self.learning_ai.agent.epsilon:.3f}",
                'Memory': len(self.learning_ai.agent.memory)
            })
            
            # Save periodically
            if (episode + 1) % save_interval == 0:
                self.save_progress(episode + 1)
        
        # Final save
        self.save_progress(episodes)
        print(f"\n✅ Memory-optimized training complete!")
    
    def save_progress(self, episode):
        """Save training progress with minimal memory usage."""
        model_path = os.path.join(self.output_dir, f"memory_model_episode_{episode}.pth")
        self.learning_ai.save_model(model_path)
        
        # Save only essential stats
        stats = {
            'episode': episode,
            'avg_reward_last_50': np.mean(self.episode_rewards[-50:]) if self.episode_rewards else 0,
            'best_reward': max(self.episode_rewards) if self.episode_rewards else 0,
            'epsilon': self.learning_ai.agent.epsilon,
            'training_steps': self.learning_ai.agent.training_step
        }
        
        stats_path = os.path.join(self.output_dir, f"memory_stats_episode_{episode}.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)


def main():
    """Main memory-optimized training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Memory-Optimized GPU Training')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--grid-size', type=int, nargs=2, default=(15, 15), help='Grid size')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--model', type=str, help='Path to pre-trained model')
    
    args = parser.parse_args()
    
    trainer = MemoryOptimizedTrainer(
        grid_size=tuple(args.grid_size),
        device=args.device,
        model_path=args.model
    )
    
    trainer.train(episodes=args.episodes)


if __name__ == '__main__':
    main() 