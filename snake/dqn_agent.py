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


Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class SnakeDQN(nn.Module):
    """Deep Q-Network for Snake game."""
    
    def __init__(self, grid_size: Tuple[int, int], device: str = 'cuda'):
        super(SnakeDQN, self).__init__()
        self.grid_size = grid_size
        self.device = device
        
        # Input: grid_size[0] x grid_size[1] x 3 (snake, food, head)
        input_channels = 3
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Calculate flattened size after convolutions
        conv_output_size = 64 * grid_size[0] * grid_size[1]
        
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 4)  # 4 actions: up, down, left, right
        
        self.to(device)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    """Deep Q-Learning agent for Snake game."""
    
    def __init__(self, grid_size: Tuple[int, int], learning_rate: float = 0.001, 
                 gamma: float = 0.99, epsilon: float = 1.0, epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995, memory_size: int = 10000, 
                 batch_size: int = 64, device: str = 'cuda'):
        
        self.grid_size = grid_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.device = device
        
        # Networks
        self.q_network = SnakeDQN(grid_size, device)
        self.target_network = SnakeDQN(grid_size, device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Use mixed precision for faster training
        self.scaler = torch.cuda.amp.GradScaler()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay
        self.memory = deque(maxlen=memory_size)
        
        # Training stats
        self.training_step = 0
        self.episode_rewards = []
        self.episode_lengths = []
        
    def get_state_representation(self, game_state) -> torch.Tensor:
        """Convert game state to neural network input."""
        # Get actual grid size from game state
        actual_grid_size = (game_state.grid_width, game_state.grid_height)
        
        # Create 3-channel representation: snake body, food, head
        snake_channel = np.zeros(actual_grid_size, dtype=np.float32)
        food_channel = np.zeros(actual_grid_size, dtype=np.float32)
        head_channel = np.zeros(actual_grid_size, dtype=np.float32)
        
        # Snake body (excluding head)
        for segment in game_state.get_snake_body()[1:]:
            x, y = segment
            if 0 <= x < actual_grid_size[0] and 0 <= y < actual_grid_size[1]:
                snake_channel[y, x] = 1.0
        
        # Food
        fx, fy = game_state.food
        if 0 <= fx < actual_grid_size[0] and 0 <= fy < actual_grid_size[1]:
            food_channel[fy, fx] = 1.0
        
        # Snake head
        head = game_state.get_snake_head()
        hx, hy = head
        if 0 <= hx < actual_grid_size[0] and 0 <= hy < actual_grid_size[1]:
            head_channel[hy, hx] = 1.0
        
        # Stack channels and add batch dimension
        state = np.stack([snake_channel, food_channel, head_channel], axis=0)
        return torch.FloatTensor(state).unsqueeze(0).to(self.device)
    
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
        self.memory.append(Experience(state, action, reward, next_state, done))
    
    def replay(self):
        """Train the network on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        batch = Experience(*zip(*batch))
        
        # Convert to tensors
        states = torch.cat(batch.state)
        actions = torch.LongTensor(batch.action).to(self.device)
        rewards = torch.FloatTensor(batch.reward).to(self.device)
        next_states = torch.cat(batch.next_state)
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
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # Update target network periodically
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