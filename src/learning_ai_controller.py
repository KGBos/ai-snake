import torch
import numpy as np
from typing import Tuple, Optional
from .models import GameState
from .dqn_agent import DQNAgent
import logging
import os
from datetime import datetime

# Set up file logging for post-game analysis
def setup_game_logging():
    """Set up logging to file for post-game analysis."""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create unique log file name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/game_session_{timestamp}.log"
    
    # Configure logging to write to file only (no console output)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, mode='w')
            # Removed StreamHandler to eliminate console output
        ]
    )
    
    logger = logging.getLogger('GameAnalysis')
    logger.info(f"=== GAME SESSION STARTED ===")
    logger.info(f"Log file: {log_filename}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    return logger, log_filename

# Global logger for the game session
game_logger, log_file_path = setup_game_logging()

class LearningAIController:
    """Learning AI controller that uses DQN to make decisions."""
    
    def __init__(self, grid_size: Tuple[int, int], device: str = 'cuda', 
                 model_path: Optional[str] = None, training: bool = True):
        self.grid_size = grid_size
        self.device = device
        self.training = training
        
        # Initialize DQN agent
        self.agent = DQNAgent(device=device)
        
        # Load pre-trained model if available
        if model_path:
            self.agent.load_model(model_path)
            game_logger.info(f"Loaded pre-trained model from {model_path}")
        else:
            # Try to load auto-saved model if no specific model path provided
            auto_save_filename = "snake_dqn_model_auto.pth"
            if os.path.exists(auto_save_filename):
                self.agent.load_model(auto_save_filename)
                game_logger.info(f"Auto-loaded previous training model from {auto_save_filename}")
        
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
        
        # Action mapping: 0=up, 1=down, 2=left, 3=right
        self.action_to_direction = {
            0: (0, -1),   # Up
            1: (0, 1),    # Down
            2: (-1, 0),   # Left
            3: (1, 0)     # Right
        }
        
        self.episode_stats = []  # Store per-episode stats for later review
        
        game_logger.info(f"Learning AI initialized with grid size {grid_size}, device {device}, training={training}")
    
    def get_action(self, game_state: GameState) -> Tuple[int, int]:
        """Get the next action from the DQN agent."""
        self.step_count += 1
        
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
        
        # Track episode statistics
        self.total_reward_this_episode += reward
        
        # Check if food was eaten
        if game_state.score > getattr(self, 'last_score', 0):
            self.food_eaten_this_episode += 1
        
        # Check if snake died
        if done:
            self.deaths_this_episode += 1
        
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
        
        # Train the agent every step (more aggressive learning)
        self.agent.replay()
        
        # Update last state and score
        self.last_state = current_state
        self.last_score = game_state.score
    
    def record_episode_end(self, final_score: int, death_type: Optional[str] = None):
        """Record the end of an episode."""
        if self.training:
            self.agent.episode_rewards.append(self.current_episode_reward)
            self.agent.episode_lengths.append(self.current_episode_length)
            
            # Store summary stats for this episode
            self.episode_stats.append({
                'final_score': final_score,
                'episode_length': self.current_episode_length,
                'total_reward': self.current_episode_reward,
                'food_eaten': self.food_eaten_this_episode,
                'deaths': self.deaths_this_episode,
                'memory_size': len(self.agent.memory),
                'epsilon': self.agent.epsilon,
                'death_type': death_type if death_type else 'unknown'
            })
            
            # Log a CSV-style summary at the end of each episode
            game_logger.info(f"EPISODE {len(self.episode_stats)},Score={final_score},Length={self.current_episode_length},Reward={self.current_episode_reward:.2f},Food={self.food_eaten_this_episode},Deaths={self.deaths_this_episode},Memory={len(self.agent.memory)},Epsilon={self.agent.epsilon:.3f},DeathType={death_type if death_type else 'unknown'}")
            
            self.current_episode_reward = 0
            self.current_episode_length = 0
            self.last_state = None
            self.last_action = None
            self.step_count = 0
            self.food_eaten_this_episode = 0
            self.deaths_this_episode = 0
            self.total_reward_this_episode = 0
            self.last_score = 0
    
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
        self.last_direction = None  # Track last direction for distance calculation
        self.starvation_threshold = starvation_threshold
        self.last_positions = []  # For oscillation detection
        self.food_move_counter = 0  # For efficiency bonus
        self.last_path_available = None  # For path-availability bonus
        
        # Enhanced logging
        self.step_count = 0
        self.reward_breakdown = {}
        
        # Load reward parameters from config or use defaults
        if config and 'learning' in config:
            learning_config = config['learning']
            self.food_reward = learning_config.get('food_reward', 100.0)  # High food reward
            self.death_penalty = learning_config.get('death_penalty', -50.0)  # Strong death penalty
            self.move_penalty = learning_config.get('move_penalty', 0.01)  # Small move penalty
            self.distance_reward_weight = learning_config.get('distance_reward_weight', 2.0)  # Increased distance reward
            self.distance_penalty_weight = learning_config.get('distance_penalty_weight', 0.1)
            self.efficiency_bonus = learning_config.get('efficiency_bonus', 10.0)
            self.survival_bonus = learning_config.get('survival_bonus', 0.1)
            self.oscillation_penalty = learning_config.get('oscillation_penalty', 0.1)
            self.path_efficiency_bonus = learning_config.get('path_efficiency_bonus', 5.0)
            self.space_utilization_bonus = learning_config.get('space_utilization_bonus', 0.5)
            self.direction_reversal_penalty = learning_config.get('direction_reversal_penalty', 0.2)
        else:
            # Default values for natural learning
            self.food_reward = 100.0
            self.death_penalty = -50.0
            self.move_penalty = 0.01
            self.distance_reward_weight = 2.0
            self.distance_penalty_weight = 0.1
            self.efficiency_bonus = 10.0
            self.survival_bonus = 0.1
            self.oscillation_penalty = 0.1
            self.path_efficiency_bonus = 5.0
            self.space_utilization_bonus = 0.5
            self.direction_reversal_penalty = 0.2
        
        game_logger.info(f"RewardCalculator initialized with food_reward={self.food_reward}, death_penalty={self.death_penalty}")
    
    def calculate_distance_to_food(self, head_pos, food_pos):
        """Calculate Manhattan distance to food."""
        return abs(head_pos[0] - food_pos[0]) + abs(head_pos[1] - food_pos[1])
    
    def calculate_reward(self, game_state: GameState, done: bool) -> float:
        """Calculate reward with advanced shaping for accelerated learning."""
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
        
        # 5. STARVATION PENALTY (Prevent infinite loops)
        if self.moves_without_food > self.starvation_threshold:
            starvation_penalty = -0.1 * (self.moves_without_food - self.starvation_threshold)
            reward += starvation_penalty
            self.reward_breakdown['starvation_penalty'] = starvation_penalty
        
        # 6. SMALL MOVE PENALTY (Encourage efficiency)
        reward -= self.move_penalty
        self.reward_breakdown['move_penalty'] = -self.move_penalty
        
        # 7. OSCILLATION PENALTY (penalize revisiting positions)
        self.last_positions.append(current_head)
        if len(self.last_positions) > 8:
            self.last_positions.pop(0)
        if self.last_positions.count(current_head) > 1:
            reward -= self.oscillation_penalty
            self.reward_breakdown['oscillation_penalty'] = -self.oscillation_penalty
        
        # 8. PATH AVAILABILITY BONUS (bonus if path to food exists)
        from src.ai_controller import AIController
        ai = AIController()
        path_exists = ai.path_exists(current_head, current_food, set(game_state.get_snake_body()[1:]), game_state.grid_width, game_state.grid_height)
        if path_exists:
            reward += self.path_efficiency_bonus
            self.reward_breakdown['path_efficiency_bonus'] = self.path_efficiency_bonus
        else:
            reward -= self.path_efficiency_bonus
            self.reward_breakdown['path_blocked_penalty'] = -self.path_efficiency_bonus
        
        # 9. DANGEROUS MOVE PENALTY (close to wall or self)
        x, y = current_head
        danger = (
            x == 0 or x == game_state.grid_width - 1 or
            y == 0 or y == game_state.grid_height - 1 or
            current_head in list(game_state.snake)[1:]
        )
        if danger:
            reward -= self.direction_reversal_penalty
            self.reward_breakdown['danger_penalty'] = -self.direction_reversal_penalty
        
        # Update tracking variables
        self.last_score = game_state.score
        self.last_head_pos = current_head
        self.last_food_pos = current_food
        
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

def analyze_game_log(log_file_path: str) -> dict:
    """Analyze the game log file and return comprehensive statistics."""
    if not os.path.exists(log_file_path):
        return {"error": "Log file not found"}
    
    analysis = {
        "total_steps": 0,
        "total_episodes": 0,
        "total_food_eaten": 0,
        "total_deaths": 0,
        "total_safety_interventions": 0,
        "episode_details": [],
        "learning_progress": [],
        "action_distribution": {},
        "reward_breakdown": {},
        "safety_interventions": [],
        "final_stats": {}
    }
    
    with open(log_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Parse episode end entries (new CSV format)
            if "EPISODE" in line and "Score=" in line and "," in line:
                try:
                    # Extract the CSV part after "INFO - "
                    if " - INFO - " in line:
                        csv_part = line.split(" - INFO - ")[1]
                    else:
                        csv_part = line
                    
                    # Parse: "EPISODE X,Score=X,Length=X,Reward=X,Food=X,Deaths=X,Memory=X,Epsilon=X,DeathType=X"
                    parts = csv_part.split(",")
                    
                    episode_data = {}
                    for part in parts:
                        if "=" in part:
                            key, value = part.split("=")
                            if key == "Score":
                                episode_data["final_score"] = int(value)
                            elif key == "Length":
                                episode_data["episode_length"] = int(value)
                            elif key == "Reward":
                                episode_data["total_reward"] = float(value)
                            elif key == "Food":
                                episode_data["food_eaten"] = int(value)
                            elif key == "Deaths":
                                episode_data["deaths"] = int(value)
                            elif key == "Memory":
                                episode_data["memory_size"] = int(value)
                            elif key == "Epsilon":
                                episode_data["epsilon"] = float(value)
                            elif key == "DeathType":
                                episode_data["death_type"] = value
                    
                    # Extract episode number from "EPISODE X"
                    episode_part = parts[0]
                    episode_num = int(episode_part.split("EPISODE")[1].strip())
                    episode_data["episode_number"] = episode_num
                    
                    analysis["episode_details"].append(episode_data)
                    analysis["total_episodes"] += 1
                    analysis["total_food_eaten"] += episode_data.get("food_eaten", 0)
                    analysis["total_deaths"] += episode_data.get("deaths", 0)
                    
                except Exception as e:
                    print(f"Error parsing episode line: {line} - {e}")
                    continue
    
    # Calculate final statistics
    if analysis["episode_details"]:
        scores = [ep["final_score"] for ep in analysis["episode_details"]]
        rewards = [ep["total_reward"] for ep in analysis["episode_details"]]
        food_per_episode = [ep["food_eaten"] for ep in analysis["episode_details"]]
        deaths_per_episode = [ep["deaths"] for ep in analysis["episode_details"]]
        
        # Count death types
        death_types = {}
        for ep in analysis["episode_details"]:
            death_type = ep.get("death_type", "unknown")
            death_types[death_type] = death_types.get(death_type, 0) + 1
        
        analysis["final_stats"] = {
            "average_score": sum(scores) / len(scores),
            "best_score": max(scores),
            "worst_score": min(scores),
            "average_reward": sum(rewards) / len(rewards),
            "best_reward": max(rewards),
            "worst_reward": min(rewards),
            "average_food_per_episode": sum(food_per_episode) / len(food_per_episode),
            "average_deaths_per_episode": sum(deaths_per_episode) / len(deaths_per_episode),
            "total_episodes": len(analysis["episode_details"]),
            "learning_progress": "improving" if len(scores) > 1 and scores[-1] > scores[0] else "stable",
            "death_types": death_types
        }
    
    return analysis

def print_game_analysis(log_file_path: str):
    """Print a comprehensive analysis of the game session."""
    analysis = analyze_game_log(log_file_path)
    
    if "error" in analysis:
        print(f"Error analyzing log: {analysis['error']}")
        return
    
    print("\n" + "="*80)
    print("🎮 GAME SESSION ANALYSIS")
    print("="*80)
    
    print(f"📊 OVERALL STATISTICS:")
    print(f"   Total Steps: {analysis['total_steps']:,}")
    print(f"   Total Episodes: {analysis['total_episodes']}")
    print(f"   Total Food Eaten: {analysis['total_food_eaten']}")
    print(f"   Total Deaths: {analysis['total_deaths']}")
    print(f"   Total Safety Interventions: {analysis['total_safety_interventions']}")
    
    if analysis["final_stats"]:
        stats = analysis["final_stats"]
        print(f"\n📈 PERFORMANCE ANALYSIS:")
        print(f"   Average Score: {stats['average_score']:.2f}")
        print(f"   Best Score: {stats['best_score']}")
        print(f"   Worst Score: {stats['worst_score']}")
        print(f"   Average Reward: {stats['average_reward']:.2f}")
        print(f"   Best Reward: {stats['best_reward']:.2f}")
        print(f"   Average Food per Episode: {stats['average_food_per_episode']:.2f}")
        print(f"   Average Deaths per Episode: {stats['average_deaths_per_episode']:.2f}")
        print(f"   Learning Progress: {stats['learning_progress'].upper()}")
        
        # Display death type breakdown
        if 'death_types' in stats:
            print(f"\n💀 DEATH TYPE BREAKDOWN:")
            total_episodes = stats['total_episodes']
            for death_type, count in sorted(stats['death_types'].items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_episodes * 100) if total_episodes > 0 else 0
                print(f"   {death_type.capitalize()}: {count} episodes ({percentage:.1f}%)")
    
    if analysis["action_distribution"]:
        print(f"\n🎯 ACTION DISTRIBUTION:")
        total_actions = sum(analysis["action_distribution"].values())
        action_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
        for action, count in sorted(analysis["action_distribution"].items()):
            percentage = (count / total_actions * 100) if total_actions > 0 else 0
            print(f"   {action_names.get(action, f'Action_{action}')}: {count} ({percentage:.1f}%)")
    
    if analysis["episode_details"]:
        print(f"\n📋 EPISODE DETAILS:")
        for i, episode in enumerate(analysis["episode_details"][-5:], 1):  # Show last 5 episodes
            print(f"   Episode {episode['episode_number']}: Score={episode['final_score']}, "
                  f"Food={episode['food_eaten']}, Deaths={episode['deaths']}, "
                  f"Reward={episode['total_reward']:.2f}")
    
    if analysis["safety_interventions"]:
        print(f"\n🛡️ RECENT SAFETY INTERVENTIONS:")
        for intervention in analysis["safety_interventions"][-3:]:  # Show last 3
            print(f"   {intervention}")
    
    print("="*80)
    print(f"📄 Full log available at: {log_file_path}")
    print("="*80) 