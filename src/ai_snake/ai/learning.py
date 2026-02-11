import torch
import numpy as np
from typing import Tuple, Optional
from ai_snake.game.models import GameState
from ai_snake.ai.dqn import DQNAgent
import logging
import os
from datetime import datetime
from ai_snake.ai.rule_based import AIController
import json

# ---------------------------------------------------------------------------
# Lazy game-session logging — only initialized when first accessed, so merely
# importing this module has no side effects (no log files created, no root
# logger mutation).
# ---------------------------------------------------------------------------

# Lazy logging setup
_game_logger: Optional[logging.Logger] = None

def get_game_logger() -> logging.Logger:
    """Return the game-session logger, initializing on first call."""
    global _game_logger
    if _game_logger is None:
        from ai_snake.utils.logging_utils import setup_logging
        import glob
        setup_logging(log_to_file=True, log_to_console=False, log_level='INFO', log_name='game_session', json_mode=False)
        _game_logger = logging.getLogger('GameAnalysis')
        _game_logger.setLevel(logging.INFO)
        _game_logger.propagate = True
        
        log_files = glob.glob(os.path.join('logs', 'game_session_*.log'))
        log_file = max(log_files, key=os.path.getmtime) if log_files else "unknown"
        _game_logger.info(f"=== GAME SESSION STARTED ===\nLog file: {log_file}\nTimestamp: {datetime.now()}")
        _game_logger.info("=" * 60)
    return _game_logger

def get_log_file_path() -> Optional[str]:
    """Return the log file path."""
    import glob
    log_files = glob.glob(os.path.join('logs', 'game_session_*.log'))
    return max(log_files, key=os.path.getmtime) if log_files else None

class LearningAIController:
    """Learning AI controller that uses DQN to make decisions."""
    
    def __init__(self, grid_size: Tuple[int, int], device: Optional[str] = None, 
                 model_path: Optional[str] = None, training: bool = True, train_frequency: int = 4,
                 wandb_logger=None):
        self.grid_size = grid_size
        self.wandb_logger = wandb_logger
        
        # Auto-detect device if not specified
        if device is None:
            if torch.backends.mps.is_available():
                device = 'mps'
            elif torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
                
        self.device = device
        self.training = training
        
        # Initialize DQN agent
        self.agent = DQNAgent(device=device)
        self.reward_calculator = RewardCalculator()
        
        # Load pre-trained model if available
        if model_path:
            self.agent.load_model(model_path)
            get_game_logger().info(f"Loaded pre-trained model from {model_path}")
        else:
            # Try to load auto-saved model if no specific model path provided
            auto_save_filename = "snake_dqn_model_auto.pth"
            if os.path.exists(auto_save_filename):
                self.agent.load_model(auto_save_filename)
                get_game_logger().info(f"Auto-loaded previous training model from {auto_save_filename}")
        
        # Training state
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.last_state = None
        self.last_action = None
        
        # Enhanced logging stats
        self.step_count = 0
        self.food_eaten_this_episode = 0
        self.deaths_this_episode = 0
        self.total_reward_this_episode = 0
        self.train_frequency = train_frequency
        self.last_computed_state = None  # Cache for state representation
        self.cached_state_for_next_action = None  # Cache state from record_step for next get_action
        
        # Action mapping: 0=up, 1=down, 2=left, 3=right
        self.action_to_direction = {
            0: (0, -1),   # Up
            1: (0, 1),    # Down
            2: (-1, 0),   # Left
            3: (1, 0)     # Right
        }
        
        self.episode_stats = []  # Store per-episode stats for later review
        
        get_game_logger().info(f"Learning AI initialized with grid size {grid_size}, device {device}, training={training}")
    
    def get_action(self, game_state: GameState) -> Tuple[int, int]:
        """Get the next action from the DQN agent."""
        self.step_count += 1
        
        # Get action from agent
        # Optimization: Use cached state from record_step if available (inter-frame optimization)
        if self.cached_state_for_next_action is not None:
             state_tensor = self.cached_state_for_next_action
             self.cached_state_for_next_action = None
        else:
             state_tensor = self.agent.get_state_representation(game_state)
             
        self.last_computed_state = state_tensor
        
        action = self.agent.get_action(game_state, state_tensor=state_tensor, training=self.training)
        
        # Convert to direction
        direction = self.action_to_direction[action]
        
        # Store for training
        if self.training:
            self.last_state = state_tensor
            self.last_action = action
        
        return direction
    
    def record_step(self, game_state: GameState, reward: float, done: bool):
        """Record a training step."""
        if not self.training or self.last_state is None:
            return
        if hasattr(self, 'episode_done') and self.episode_done:
            return
            
        # Optimization: Calculate state once and reuse for reward calculation
        current_state = self.agent.get_state_representation(game_state)
        
        # OPTIMIZATION: Pass cached state to reward calculator to avoid re-running BFS
        # We need to extract the path_available flag (index 22 in the feature vector)
        # The tensor is [1, 27], so we want [0, 22]
        path_available = None
        if current_state is not None:
            # Check if we can extract path_available from tensor to avoid BFS
            # State size is 27, path_available is at index 22
            # enhanced state: [..., path_available(22), ...]
            try:
                path_available = bool(current_state[0, 22].item() > 0.5)
            except Exception:
                pass

        # Track episode statistics
        self.total_reward_this_episode += reward
        # Check if food was eaten
        if game_state.score > getattr(self, 'last_score', 0):
            self.food_eaten_this_episode += 1
        # Check if snake died
        if done:
            self.deaths_this_episode += 1
            self.episode_done = True
        # Store experience
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
        
        if self.step_count % self.train_frequency == 0:
            loss, mean_q = self.agent.replay()
            if self.wandb_logger and loss != 0:
                self.wandb_logger.log({
                    "train/loss": loss,
                    "train/mean_q": mean_q,
                    "train/epsilon": self.agent.epsilon,
                    "train/buffer_size": len(self.agent.memory)
                }, step=self.agent.training_step)
            
        # Update last state and score
        self.last_state = current_state
        self.last_score = game_state.score
        
        # caching for next frame
        self.cached_state_for_next_action = current_state
    
    def record_episode_end(self, final_score: int, death_type: Optional[str] = None):
        """Record the end of an episode."""
        # Always reset episode_done flag
        self.episode_done = False
        if self.training:
            self.agent.episode_rewards.append(self.current_episode_reward)
            self.agent.episode_lengths.append(self.current_episode_length)
            # Store summary stats for this episode
            episode_data = {
                'event': 'episode_end',
                'final_score': final_score,
                'episode_length': self.current_episode_length,
                'total_reward': self.current_episode_reward,
                'food_eaten': self.food_eaten_this_episode,
                'deaths': self.deaths_this_episode,
                'memory_size': len(self.agent.memory),
                'epsilon': self.agent.epsilon,
                'death_type': death_type if death_type else 'unknown'
            }
            self.episode_stats.append(episode_data)
            # Log a CSV-style summary at the end of each episode
            get_game_logger().info(f"EPISODE {len(self.episode_stats)},Score={final_score},Length={self.current_episode_length},Reward={self.current_episode_reward:.2f},Food={self.food_eaten_this_episode},Deaths={self.deaths_this_episode},Memory={len(self.agent.memory)},Epsilon={self.agent.epsilon:.3f},DeathType={death_type if death_type else 'unknown'}")
            # Log a JSON-structured entry for agent analysis
            try:
                get_game_logger().info(json.dumps(episode_data))
            except Exception as e:
                logging.getLogger(__name__).warning(f"Failed to log JSON episode data: {e}")
            
            # WandB Logging
            if self.wandb_logger:
                self.wandb_logger.log({
                    "game/score": final_score,
                    "game/length": self.current_episode_length,
                    "game/reward": self.current_episode_reward,
                    "game/food": self.food_eaten_this_episode,
                    "game/efficiency": self.current_episode_length / max(1, self.food_eaten_this_episode),
                    "game/death_type": death_type if death_type else "unknown"
                })
                
                # Log Replay Buffer Histogram occasionally
                if len(self.agent.episode_rewards) % 50 == 0 and len(self.agent.memory) > 0:
                    # Sample rewards from memory
                    sample_size = min(1000, len(self.agent.memory))
                    sample = list(self.agent.memory)[:sample_size] # Simple slice or random sample
                    rewards = [e.reward for e in sample]
                    self.wandb_logger.log_histogram("params/buffer_rewards", rewards)
            # Always reset episode stats
            self.current_episode_reward = 0
            self.current_episode_length = 0
            self.last_state = None
            self.last_action = None
            self.step_count = 0
            self.food_eaten_this_episode = 0
            self.deaths_this_episode = 0
            self.total_reward_this_episode = 0
            self.last_score = 0
            # Clear cache on episode end
            self.cached_state_for_next_action = None
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        self.agent.save_model(filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        self.agent.load_model(filepath)
    
    def get_stats(self) -> dict:
        """Get training statistics."""
        stats = self.agent.get_stats()
        return stats
    
    def set_training_mode(self, training: bool):
        """Set training mode."""
        self.training = training


class RewardCalculator:
    """Calculate rewards for the learning agent with advanced reward shaping."""
    
    def __init__(self, config=None, starvation_threshold: int = 50):
        self.last_food_eaten = 0
        self.last_score = 0
        self.last_head_pos = None
        self.last_food_pos = None
        self.moves_without_food = 0
        self.last_positions = []
        self.food_move_counter = 0
        self.starvation_threshold = starvation_threshold
        # Enhanced logging
        self.step_count = 0
        self.reward_breakdown = {}
        
        # Load params with defaults
        lc = config.get('learning', {}) if config else {}
        defaults = {
            'food_reward': 100.0, 'death_penalty': -50.0, 'move_penalty': 0.01,
            'distance_reward_weight': 2.0, 'distance_penalty_weight': 0.1,
            'efficiency_bonus': 10.0, 'survival_bonus': 0.1, 'oscillation_penalty': 0.1,
            'path_efficiency_bonus': 5.0, 'space_utilization_bonus': 0.5,
            'direction_reversal_penalty': 0.2
        }
        for k, v in defaults.items():
            setattr(self, k, lc.get(k, v))
            
        get_game_logger().info(f"RewardCalculator initialized: {defaults}")
    
    def calculate_distance_to_food(self, head_pos, food_pos):
        """Calculate Manhattan distance to food."""
        return abs(head_pos[0] - food_pos[0]) + abs(head_pos[1] - food_pos[1])
    
    def calculate_reward(self, game_state: GameState, done: bool, path_available: Optional[bool] = None) -> float:
        """Calculate reward with advanced shaping for accelerated learning.
        
        Args:
            game_state: Current game state
            done: Whether game is over
            path_available: Optional optimization to avoid re-calculating BFS if already known
        """
        self.step_count += 1
        reward = 0.0
        current_head = game_state.get_snake_head()
        current_food = game_state.food
        
        # Reset reward breakdown for this step
        self.reward_breakdown = {}
        
        # 1. FOOD REWARD (Primary objective)
        if game_state.score > self.last_score:
            reward += self.food_reward
            self.reward_breakdown['food_reward'] = self.food_reward
            self.last_food_eaten += 1
            self.moves_without_food = 0
            # Efficiency bonus: fewer moves to food
            if self.food_move_counter > 0 and self.food_move_counter <= 10:
                reward += self.efficiency_bonus
                self.reward_breakdown['efficiency_bonus'] = self.efficiency_bonus
            self.food_move_counter = 0
        else:
            self.moves_without_food += 1
            self.food_move_counter += 1
        
        # 2. DEATH PENALTY (Strong penalty for dying)
        if done:
            reward += self.death_penalty
            self.reward_breakdown['death_penalty'] = self.death_penalty
        
        # 3. DISTANCE-BASED REWARD (encourage moving toward food)
        if self.last_head_pos is not None:
            old_distance = self.calculate_distance_to_food(self.last_head_pos, current_food)
            new_distance = self.calculate_distance_to_food(current_head, current_food)
            if new_distance < old_distance:
                distance_reward = self.distance_reward_weight
                reward += distance_reward
                self.reward_breakdown['distance_reward'] = distance_reward
            elif new_distance > old_distance:
                distance_penalty = -self.distance_penalty_weight
                reward += distance_penalty
                self.reward_breakdown['distance_penalty'] = distance_penalty
            else:
                stay_reward = 0.05
                reward += stay_reward
                self.reward_breakdown['stay_reward'] = stay_reward
        
        # 4. SURVIVAL BONUS (small reward for each move)
        reward += self.survival_bonus
        self.reward_breakdown['survival_bonus'] = self.survival_bonus
        
        return reward
    
    def reset(self):
        """Reset the reward calculator."""
        self.last_food_eaten = 0
        self.last_score = 0
        self.last_head_pos = None
        self.last_food_pos = None
        self.moves_without_food = 0
        self.last_direction = None  # Reset direction tracking
        self.last_positions = []
        self.food_move_counter = 0
        self.last_path_available = None

# Removed redundant analyze_game_log and print_game_analysis functions to reduce LOC.
# GameController.print_learning_report provides equivalent in-memory stats. 