import torch
import numpy as np
from typing import Tuple, Optional
from .models import GameState
from .dqn_agent import DQNAgent


class LearningAIController:
    """Learning AI controller that uses DQN to make decisions."""
    
    def __init__(self, grid_size: Tuple[int, int], device: str = 'cuda', 
                 model_path: Optional[str] = None, training: bool = True):
        self.grid_size = grid_size
        self.device = device
        self.training = training
        
        # Initialize DQN agent
        self.agent = DQNAgent(grid_size, device=device)
        
        # Load pre-trained model if available
        if model_path:
            self.agent.load_model(model_path)
        
        # Training state
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.last_state = None
        self.last_action = None
        
        # Action mapping: 0=up, 1=down, 2=left, 3=right
        self.action_to_direction = {
            0: (0, -1),   # Up
            1: (0, 1),    # Down
            2: (-1, 0),   # Left
            3: (1, 0)     # Right
        }
    
    def get_action(self, game_state: GameState) -> Tuple[int, int]:
        """Get the next action from the DQN agent."""
        # Get action from agent
        action = self.agent.get_action(game_state, training=self.training)
        
        # Convert to direction
        direction = self.action_to_direction[action]
        
        # Store for training
        if self.training:
            self.last_state = self.agent.get_state_representation(game_state)
            self.last_action = action
        
        return direction
    
    def record_step(self, game_state: GameState, reward: float, done: bool):
        """Record a training step."""
        if not self.training or self.last_state is None:
            return
        
        # Get current state
        current_state = self.agent.get_state_representation(game_state)
        
        # Store experience
        self.agent.remember(
            self.last_state, 
            self.last_action, 
            reward, 
            current_state, 
            done
        )
        
        # Update episode stats
        self.current_episode_reward += reward
        self.current_episode_length += 1
        
        # Train the agent
        self.agent.replay()
        
        # Update last state
        self.last_state = current_state
    
    def record_episode_end(self, final_score: int):
        """Record the end of an episode."""
        if self.training:
            self.agent.episode_rewards.append(self.current_episode_reward)
            self.agent.episode_lengths.append(self.current_episode_length)
            
            # Reset episode stats
            self.current_episode_reward = 0
            self.current_episode_length = 0
            self.last_state = None
            self.last_action = None
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        self.agent.save_model(filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        self.agent.load_model(filepath)
    
    def get_stats(self) -> dict:
        """Get training statistics."""
        return self.agent.get_stats()
    
    def set_training_mode(self, training: bool):
        """Set training mode."""
        self.training = training


class RewardCalculator:
    """Calculate rewards for the learning agent."""
    
    def __init__(self):
        self.last_food_eaten = 0
        self.last_score = 0
    
    def calculate_reward(self, game_state: GameState, done: bool) -> float:
        """Calculate reward for current state."""
        reward = 0.0
        
        # Reward for eating food
        if game_state.score > self.last_score:
            reward += 10.0  # Big reward for eating food
            self.last_food_eaten += 1
        
        # Penalty for dying
        if done:
            reward -= 10.0
        
        # Small penalty for each move (encourage efficiency)
        reward -= 0.1
        
        # Bonus for survival
        if not done:
            reward += 0.1
        
        # Update last score
        self.last_score = game_state.score
        
        return reward
    
    def reset(self):
        """Reset the reward calculator."""
        self.last_food_eaten = 0
        self.last_score = 0 