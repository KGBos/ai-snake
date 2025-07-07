#!/usr/bin/env python3
"""
Training script for the Snake DQN agent.
This script runs automated training sessions and saves progress.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
from datetime import datetime

from snake.game_controller import GameController
from snake.learning_ai_controller import LearningAIController, RewardCalculator


class TrainingManager:
    """Manages the training process for the DQN agent."""
    
    def __init__(self, grid_size=(20, 20), device='cuda', model_path=None, speed=10):
        self.grid_size = grid_size
        self.device = device
        self.model_path = model_path
        self.speed = speed
        
        # Initialize learning components with optimized settings
        self.learning_ai = LearningAIController(
            grid_size=grid_size,
            device=device,
            model_path=model_path,
            training=True
        )
        
        # Optimize for GPU training
        if device == 'cuda':
            # Increase batch size for GPU
            self.learning_ai.agent.batch_size = 128
            # Increase memory size for more experiences
            self.learning_ai.agent.memory_size = 20000
            # Use faster epsilon decay
            self.learning_ai.agent.epsilon_decay = 0.999
        
        self.reward_calculator = RewardCalculator()
        
        # Training stats
        self.episode_rewards = []
        self.episode_lengths = []
        self.food_eaten_per_episode = []
        self.epsilon_history = []
        
        # Create output directory
        self.output_dir = f"training_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def train_episode(self, max_steps=1000):
        """Train for one episode."""
        # Import GameState here to avoid circular imports
        from snake.models import GameState
        
        # Create proper game state
        game_state = GameState(grid_width=self.grid_size[0], grid_height=self.grid_size[1])
        
        # Initialize game state
        game_state.reset()
        game_state.direction = (1, 0)
        game_state.food = self._generate_food(game_state)
        game_state.score = 0
        game_state.game_over = False
        
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
            
            # Train the agent
            self.learning_ai.agent.replay()
            
            episode_reward += reward
            steps += 1
            
            # Track food eaten
            if game_state.score > food_eaten * 10:  # Assuming 10 points per food
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
        # Update direction
        game_state.direction = direction
        
        # Move snake
        head = game_state.snake[0]
        new_head = (head[0] + direction[0], head[1] + direction[1])
        
        # Check wall collision
        if (new_head[0] < 0 or new_head[0] >= self.grid_size[0] or
            new_head[1] < 0 or new_head[1] >= self.grid_size[1]):
            game_state.game_over = True
            return
        
        # Check self collision
        if new_head in game_state.snake:
            game_state.game_over = True
            return
        
        # Move snake
        game_state.snake.insert(0, new_head)
        
        # Check food collision
        if new_head == game_state.food:
            game_state.score += 10
            game_state.food = self._generate_food(game_state)
        else:
            game_state.snake.pop()
    
    def train(self, episodes=1000, save_interval=100):
        """Train the agent for specified number of episodes."""
        print(f"Starting training for {episodes} episodes...")
        print(f"Device: {self.device}")
        print(f"Grid size: {self.grid_size}")
        print(f"Speed: {self.speed}")
        
        progress_bar = tqdm(range(episodes), desc="Training")
        
        for episode in progress_bar:
            reward, steps, food_eaten = self.train_episode()
            
            # Update progress bar
            avg_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
            progress_bar.set_postfix({
                'Reward': f"{reward:.1f}",
                'Avg Reward': f"{avg_reward:.1f}",
                'Epsilon': f"{self.learning_ai.agent.epsilon:.3f}",
                'Food': food_eaten
            })
            
            # Save periodically
            if (episode + 1) % save_interval == 0:
                self.save_progress(episode + 1)
        
        # Final save
        self.save_progress(episodes)
        print(f"\nTraining complete! Results saved to {self.output_dir}")
    
    def save_progress(self, episode):
        """Save training progress."""
        # Save model
        model_path = os.path.join(self.output_dir, f"model_episode_{episode}.pth")
        self.learning_ai.save_model(model_path)
        
        # Save training stats
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
        
        stats_path = os.path.join(self.output_dir, f"stats_episode_{episode}.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Create plots
        self._create_plots(episode)
    
    def _create_plots(self, episode):
        """Create training plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        
        # Moving average rewards
        if len(self.episode_rewards) >= 100:
            moving_avg = [np.mean(self.episode_rewards[max(0, i-100):i+1]) 
                         for i in range(len(self.episode_rewards))]
            axes[0, 1].plot(moving_avg)
            axes[0, 1].set_title('Moving Average Rewards (100 episodes)')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Average Reward')
        
        # Food eaten per episode
        axes[1, 0].plot(self.food_eaten_per_episode)
        axes[1, 0].set_title('Food Eaten per Episode')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Food Eaten')
        
        # Epsilon decay
        axes[1, 1].plot(self.epsilon_history)
        axes[1, 1].set_title('Epsilon Decay')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Epsilon')
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, f"training_plots_episode_{episode}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Snake DQN Agent')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--grid-size', type=int, nargs=2, default=(20, 20), help='Grid size (width height)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--model', type=str, help='Path to pre-trained model')
    parser.add_argument('--save-interval', type=int, default=100, help='Save interval')
    parser.add_argument('--speed', type=int, default=10, help='Training speed (higher = faster)')
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Create training manager
    trainer = TrainingManager(
        grid_size=tuple(args.grid_size),
        device=args.device,
        model_path=args.model,
        speed=args.speed
    )
    
    # Start training
    trainer.train(episodes=args.episodes, save_interval=args.save_interval)


if __name__ == '__main__':
    main() 