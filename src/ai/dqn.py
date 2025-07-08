import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque, namedtuple
import random
from typing import Tuple, List, Optional
import pickle
import os
import logging

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class SimpleSnakeDQN(nn.Module):
    """Simple DQN for Snake game with proper state representation."""
    
    def __init__(self, input_size: int = 8, hidden_size: int = 64, output_size: int = 4):
        super(SimpleSnakeDQN, self).__init__()
        
        # Simple feedforward network
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        # Initialize weights properly
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights to prevent explosion."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    """Simple DQN agent that actually learns."""
    
    def __init__(self, learning_rate: float = 0.001, gamma: float = 0.95, 
                 epsilon: float = 1.0, epsilon_min: float = 0.1, epsilon_decay: float = 0.995,
                 memory_size: int = 10000, batch_size: int = 32, device: str = 'cuda'):
        
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.device = device
        
        # Enhanced state: [pos_x, pos_y, food_x, food_y, dir_up, dir_down, dir_left, dir_right,
        #                  wall_dist_up, wall_dist_down, wall_dist_left, wall_dist_right,
        #                  danger_8_directions, food_dist, snake_length, path_available,
        #                  food_above, food_below, food_left, food_right]
        self.state_size = 27  # 4 + 4 + 4 + 8 + 1 + 1 + 1 + 4 = 27 total
        self.action_size = 4
        
        # Networks with larger hidden layers for better learning
        self.q_network = SimpleSnakeDQN(self.state_size, 128, self.action_size).to(device)
        self.target_network = SimpleSnakeDQN(self.state_size, 128, self.action_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay
        self.memory = deque(maxlen=memory_size)
        
        # Training stats
        self.training_step = 0
        self.episode_rewards = []
        self.episode_lengths = []
        
    def get_state_representation(self, game_state) -> torch.Tensor:
        """Get enhanced state representation with more detailed information."""
        head = game_state.get_snake_head()
        food = game_state.food
        direction = game_state.direction
        snake_body = game_state.get_snake_body()
        
        # Enhanced state vector with more information
        state = []
        
        # 1. Normalized positions (0-1 range)
        head_x = head[0] / game_state.grid_width
        head_y = head[1] / game_state.grid_height
        food_x = food[0] / game_state.grid_width
        food_y = food[1] / game_state.grid_height
        state.extend([head_x, head_y, food_x, food_y])
        
        # 2. Current direction (one-hot encoded)
        dir_up = 1.0 if direction == (0, -1) else 0.0
        dir_down = 1.0 if direction == (0, 1) else 0.0
        dir_left = 1.0 if direction == (-1, 0) else 0.0
        dir_right = 1.0 if direction == (1, 0) else 0.0
        state.extend([dir_up, dir_down, dir_left, dir_right])
        
        # 3. Distance to walls in each direction
        dist_to_wall_up = head[1] / game_state.grid_height
        dist_to_wall_down = (game_state.grid_height - 1 - head[1]) / game_state.grid_height
        dist_to_wall_left = head[0] / game_state.grid_width
        dist_to_wall_right = (game_state.grid_width - 1 - head[0]) / game_state.grid_width
        state.extend([dist_to_wall_up, dist_to_wall_down, dist_to_wall_left, dist_to_wall_right])
        
        # 4. Enhanced danger detection (8 directions)
        dangers = []
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (1, -1), (-1, 1), (1, 1)]:
            danger = self._is_dangerous_direction(game_state, head, dx, dy)
            dangers.append(danger)
        state.extend(dangers)
        
        # 5. Distance to food (normalized)
        manhattan_dist = abs(head[0] - food[0]) + abs(head[1] - food[1])
        normalized_dist = manhattan_dist / (game_state.grid_width + game_state.grid_height)
        state.append(normalized_dist)
        
        # 6. Snake length (normalized)
        normalized_length = len(snake_body) / (game_state.grid_width * game_state.grid_height)
        state.append(normalized_length)
        
        # 7. Path to food availability (simplified)
        path_available = self._can_reach_food(game_state, head, food, snake_body)
        state.append(1.0 if path_available else 0.0)
        
        # 8. Food direction indicators
        food_above = 1.0 if food[1] < head[1] else 0.0
        food_below = 1.0 if food[1] > head[1] else 0.0
        food_left = 1.0 if food[0] < head[0] else 0.0
        food_right = 1.0 if food[0] > head[0] else 0.0
        state.extend([food_above, food_below, food_left, food_right])
        
        return torch.FloatTensor(state).unsqueeze(0).to(self.device)
    
    def _is_dangerous_direction(self, game_state, head, dx, dy) -> float:
        """Check if moving in a specific direction is dangerous."""
        new_head = (head[0] + dx, head[1] + dy)
        
        # Wall collision
        if not (0 <= new_head[0] < game_state.grid_width and 0 <= new_head[1] < game_state.grid_height):
            return 1.0
        
        # Self collision
        if new_head in game_state.get_snake_body()[1:]:
            return 1.0
        
        return 0.0
    
    def _can_reach_food(self, game_state, head, food, snake_body) -> bool:
        """Simple check if there's a path to food."""
        # Basic check: if food is reachable without hitting walls or snake
        if food in snake_body:
            return False
        
        # Check if food is in a valid position
        if not (0 <= food[0] < game_state.grid_width and 0 <= food[1] < game_state.grid_height):
            return False
        
        return True
    
    def get_action(self, game_state, training: bool = True) -> int:
        """Get action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            # Random action
            return random.randint(0, 3)
        
        # Get Q-values from network
        state = self.get_state_representation(game_state)
        with torch.no_grad():
            q_values = self.q_network(state)
        
        # Return best action
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        # Convert state tensors to CPU for storage
        if isinstance(state, torch.Tensor):
            state = state.cpu()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu()
        
        self.memory.append(Experience(state, action, reward, next_state, done))
    
    def replay(self):
        """Train the network on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        batch = Experience(*zip(*batch))
        
        # Convert to tensors and move to device
        states = torch.cat([s.to(self.device) for s in batch.state])
        actions = torch.LongTensor(batch.action).to(self.device)
        rewards = torch.FloatTensor(batch.reward).to(self.device)
        next_states = torch.cat([s.to(self.device) for s in batch.next_state])
        dones = torch.BoolTensor(batch.done).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values (from target network)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Update target network
        self.training_step += 1
        if self.training_step % 100 == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths
        }, filepath)
        logging.info(f"Model saved to {filepath} (epsilon={self.epsilon:.3f}, training_step={self.training_step})")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.training_step = checkpoint['training_step']
            self.episode_rewards = checkpoint['episode_rewards']
            self.episode_lengths = checkpoint['episode_lengths']
            print(f"Loaded model from {filepath}")
            logging.info(f"Model loaded from {filepath} (epsilon={self.epsilon:.3f}, training_step={self.training_step})")
    
    def get_stats(self) -> dict:
        """Get training statistics."""
        if not self.episode_rewards:
            return {}
        
        return {
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'memory_size': len(self.memory),
            'avg_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            'avg_length': np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0,
            'best_reward': max(self.episode_rewards) if self.episode_rewards else 0
        } 