import os
import sys
# Set SDL_VIDEODRIVER to 'dummy' for true headless mode (no window)
if '--headless' in sys.argv or os.environ.get('SNAKE_HEADLESS') == '1':
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
import pygame
from typing import Optional, Tuple
from ai_snake.game.models import GameState
from ai_snake.render.renderer import GameRenderer
from ai_snake.render.headless import HeadlessRenderer
from ai_snake.render.web import WebRenderer
from ai_snake.render.base import BaseRenderer
from ai_snake.ai.rule_based import AIController
from ai_snake.ai.learning import LearningAIController, RewardCalculator, print_game_analysis, log_file_path
from ai_snake.config.config import HIGH_SCORE_FILE
from ai_snake.config.loader import load_config
import logging
from ai_snake.game.input_handler import InputHandler
from ai_snake.game.state_manager import GameStateManager
from ai_snake.ai.manager import AIManager
from ai_snake.game.leaderboard_service import LeaderboardService

logger = logging.getLogger(__name__)

class GameController:
    """Manages the game loop and coordinates between game state, AI, and rendering."""
    
    EPISODE_ID_FILE = "episode_id.txt"

    def _load_episode_id(self):
        if os.path.exists(self.EPISODE_ID_FILE):
            try:
                with open(self.EPISODE_ID_FILE, 'r') as f:
                    return int(f.read().strip())
            except Exception:
                return 1
        return 1

    def _save_episode_id(self, episode_id):
        try:
            with open(self.EPISODE_ID_FILE, 'w') as f:
                f.write(str(episode_id))
        except Exception:
            pass

    def __init__(self, speed: int = 10, ai: bool = False, grid: Tuple[int, int] = (20, 20), 
                 nes_mode: bool = False, ai_tracing: bool = False, auto_advance: bool = False,
                 learning_ai: bool = False, model_path: Optional[str] = None, headless: bool = False, web: bool = False, starvation_threshold: Optional[int] = None):
        self.headless = headless
        self.web = web
        if self.headless:
            self.renderer = HeadlessRenderer()
            self.clock = None
        elif self.web:
            self.renderer = WebRenderer()
            self.clock = None
        else:
            self.renderer = GameRenderer(nes_mode=nes_mode, headless=headless)
            self.clock = pygame.time.Clock()
        self.state_manager = GameStateManager(grid[0], grid[1])
        self.ai_manager = AIManager(grid_size=grid, ai_tracing=ai_tracing, learning_ai=learning_ai, model_path=model_path)
        self.model_path = model_path  # Store model_path for info area
        self.starvation_threshold = starvation_threshold if starvation_threshold is not None else 50
        
        # Load config for reward system
        config = load_config('config/config.yaml')
        
        self.speed = speed
        self.ai = ai
        self.learning_ai = learning_ai
        self.ai_tracing = ai_tracing
        self.auto_advance = auto_advance
        self.high_score = self.load_high_score()
        self.state_manager.game_state.high_score = self.high_score
        
        # Training stats
        self.episode_count = self._load_episode_id()  # Persistent episode ID
        self.total_food_eaten = 0
        
        # Debug mode for learning observation
        self.debug_learning = True  # Always show learning process
        
        self.death_type = None
        
        self.input_handler = InputHandler(self)
        self.leaderboard_service = LeaderboardService()
        self.manual_teaching_mode = False
    
    def load_high_score(self) -> int:
        """Load high score from file."""
        if os.path.exists(HIGH_SCORE_FILE):
            try:
                with open(HIGH_SCORE_FILE, 'r') as f:
                    return int(f.read().strip())
            except (ValueError, IOError):
                return 0
        return 0
    
    def save_high_score(self):
        """Save high score to file."""
        try:
            with open(HIGH_SCORE_FILE, 'w') as f:
                f.write(str(self.high_score))
        except IOError:
            pass  # Silently fail if we can't save
    
    def update(self):
        """Update game state for one frame."""
        if self.headless:
            # No rendering, no event handling
            if self.ai:
                self.ai_manager.make_move(self.state_manager.game_state)
            elif self.learning_ai:
                if getattr(self, 'manual_teaching_mode', False):
                    pass
                else:
                    direction = self.ai_manager.get_action(self.state_manager.game_state)
                    if direction is not None:
                        self.state_manager.game_state.set_direction(direction, force=True)
            if self.learning_ai and self.ai_manager.learning_ai_controller:
                reward = self.ai_manager.learning_ai_controller.reward_calculator.calculate_reward(self.state_manager.game_state, self.state_manager.game_state.game_over)
                # STARVATION DEATH LOGIC
                if self.ai_manager.learning_ai_controller.reward_calculator.moves_without_food >= self.ai_manager.learning_ai_controller.reward_calculator.starvation_threshold:
                    self.state_manager.game_state.set_starvation_death()
                self.ai_manager.record_step(self.state_manager.game_state, reward, self.state_manager.game_state.game_over)
                self.current_reward = reward
                if hasattr(self, 'step_count'):
                    self.step_count += 1
                else:
                    self.step_count = 1
            old_head = self.state_manager.game_state.get_snake_head()
            self.state_manager.move_snake()
            new_head = self.state_manager.game_state.get_snake_head()
            current_time = 0
            self.state_manager.game_state.check_collision(current_time)
            # Set death type for wall/self collision
            if self.state_manager.game_state.game_over and self.death_type is None:
                self.state_manager.game_state.set_other_death()
            if self.state_manager.game_state.score > getattr(self, 'high_score', 0):
                self.high_score = self.state_manager.game_state.score
                self.state_manager.game_state.high_score = self.high_score
        else:
            if self.ai:
                logger.debug('Rule-based AI making move.')
                self.ai_manager.make_move(self.state_manager.game_state)
            elif self.learning_ai:
                # Check if in manual teaching mode
                if getattr(self, 'manual_teaching_mode', False):
                    # Manual input overrides AI in teaching mode
                    logger.debug('Manual teaching mode - AI disabled for manual input')
                else:
                    logger.debug('Learning AI making move.')
                    direction = self.ai_manager.get_action(self.state_manager.game_state)
                    if direction is not None:
                        self.state_manager.game_state.set_direction(direction, force=True)
            # Store state before move for learning
            if self.learning_ai:
                old_score = self.state_manager.game_state.score
            old_head = self.state_manager.game_state.get_snake_head()
            self.state_manager.move_snake()
            new_head = self.state_manager.game_state.get_snake_head()
            logger.debug(f'Snake moved from {old_head} to {new_head} in direction {self.state_manager.game_state.direction}')
            current_time = pygame.time.get_ticks()
            self.state_manager.game_state.check_collision(current_time)
            # Record AI events after collision detection
            if self.ai:
                self.ai_manager.check_food_eaten(self.state_manager.game_state)
            # Record learning step (even during manual teaching)
            if self.learning_ai and self.ai_manager.learning_ai_controller:
                reward = self.ai_manager.learning_ai_controller.reward_calculator.calculate_reward(self.state_manager.game_state, self.state_manager.game_state.game_over)
                # STARVATION DEATH LOGIC
                if self.ai_manager.learning_ai_controller.reward_calculator.moves_without_food >= self.ai_manager.learning_ai_controller.reward_calculator.starvation_threshold:
                    self.state_manager.game_state.set_starvation_death()
                self.ai_manager.record_step(self.state_manager.game_state, reward, self.state_manager.game_state.game_over)
                # Store current reward for display
                self.current_reward = reward
                # Track step count for episode statistics
                if hasattr(self, 'step_count'):
                    self.step_count += 1
                else:
                    self.step_count = 1
            # Update high score if needed
            if self.state_manager.game_state.score > self.high_score:
                self.high_score = self.state_manager.game_state.score
                self.state_manager.game_state.high_score = self.high_score
                logger.info(f'New high score: {self.high_score}')
            # Use death_type from GameState (set during collision detection)
            if self.state_manager.game_state.game_over and self.state_manager.game_state.death_type is None:
                self.state_manager.game_state.set_other_death()
        
    def render(self):
        """Render the current game state."""
        if self.headless:
            return
        current_time = pygame.time.get_ticks()
        # Build info dict for the info area
        info = {
            'score': self.state_manager.game_state.score,
            'high_score': self.high_score,
            'speed': self.speed,
        }
        # Determine mode and model
        if self.learning_ai:
            info['mode'] = 'Learning AI'
            if self.model_path:
                info['model'] = self.model_path
            # Add learning stats (always show, even before first death)
            stats = self.ai_manager.get_stats()
            episode_rewards = []
            if self.ai_manager.learning_ai_controller is not None:
                episode_rewards = self.ai_manager.learning_ai_controller.agent.episode_rewards
            # Always show basic stats
            info['deaths'] = self.episode_count
            info['episode'] = self.episode_count
            info['epsilon'] = stats.get('epsilon', 1.0)
            info['memory_size'] = stats.get('memory_size', 0)
            info['training_step'] = stats.get('training_step', 0)
            info['exploration_rate'] = f"{stats.get('epsilon', 1.0):.1%}"
            # Add episode progress indicator
            if episode_rewards:
                info['episode_progress'] = f"Episode {self.episode_count + 1}"
                info['total_episodes_completed'] = len(episode_rewards)
                info['episode_percentage'] = (len(episode_rewards) / max(1, len(episode_rewards))) * 100
            else:
                info['episode_progress'] = "Episode 1 (Starting)"
                info['total_episodes_completed'] = 0
                info['episode_percentage'] = 0
            # Add real-time learning info
            info['current_reward'] = getattr(self, 'current_reward', 0)
            info['teaching_mode'] = getattr(self, 'manual_teaching_mode', False)
            if self.ai_manager.learning_ai_controller is not None:
                info['learning_paused'] = not self.ai_manager.learning_ai_controller.training
            else:
                info['learning_paused'] = False
            # Performance stats (only if we have data)
            if episode_rewards:
                info['avg_reward'] = stats.get('avg_reward', 0)
                info['best_reward'] = stats.get('best_reward', 0)
                
                # Enhanced trend analysis
                if len(episode_rewards) >= 5:
                    # Recent vs previous comparison
                    recent_5 = episode_rewards[-5:]
                    recent_5_avg = sum(recent_5) / len(recent_5)
                    info['recent_5_avg'] = recent_5_avg
                    
                    if len(episode_rewards) >= 10:
                        previous_5 = episode_rewards[-10:-5]
                        previous_5_avg = sum(previous_5) / len(previous_5)
                        info['previous_5_avg'] = previous_5_avg
                        info['short_term_trend'] = recent_5_avg - previous_5_avg
                    
                    # Long-term trend (if we have enough data)
                    if len(episode_rewards) >= 20:
                        recent_20 = episode_rewards[-20:]
                        early_20 = episode_rewards[:20]
                        info['recent_20_avg'] = sum(recent_20) / len(recent_20)
                        info['early_20_avg'] = sum(early_20) / len(early_20)
                        info['long_term_improvement'] = info['recent_20_avg'] - info['early_20_avg']
                    
                    # Episode-by-episode progress
                    if len(episode_rewards) >= 3:
                        last_3 = episode_rewards[-3:]
                        info['last_3_avg'] = sum(last_3) / len(last_3)
                        info['last_episode'] = episode_rewards[-1]
                        info['second_last_episode'] = episode_rewards[-2] if len(episode_rewards) >= 2 else 0
                        info['third_last_episode'] = episode_rewards[-3] if len(episode_rewards) >= 3 else 0
                    
                    # Trend indicators
                    if len(episode_rewards) >= 3:
                        recent_trend = "📈" if episode_rewards[-1] > episode_rewards[-3] else "📉"
                        info['trend'] = recent_trend
                        
                        # More detailed trend analysis
                        if len(episode_rewards) >= 5:
                            if 'previous_5_avg' in info and recent_5_avg > info['previous_5_avg']:
                                info['trend_status'] = "IMPROVING"
                                info['trend_color'] = "green"
                            elif 'previous_5_avg' in info and recent_5_avg < info['previous_5_avg']:
                                info['trend_status'] = "DECLINING"
                                info['trend_color'] = "red"
                            else:
                                info['trend_status'] = "STABLE"
                                info['trend_color'] = "yellow"
                    
                    # Learning speed indicator
                    if len(episode_rewards) >= 10:
                        first_10_avg = sum(episode_rewards[:10]) / 10
                        last_10_avg = sum(episode_rewards[-10:]) / 10
                        learning_speed = last_10_avg - first_10_avg
                        info['learning_speed'] = learning_speed
                        
                        if learning_speed > 5:
                            info['learning_status'] = "FAST"
                        elif learning_speed > 0:
                            info['learning_status'] = "STEADY"
                        else:
                            info['learning_status'] = "SLOW"
                
                # Episode count and progress
                info['total_episodes'] = len(episode_rewards)
                info['current_episode'] = self.episode_count
                
                # Success rate (episodes with positive rewards)
                positive_episodes = sum(1 for r in episode_rewards if r > 0)
                info['success_rate'] = (positive_episodes / len(episode_rewards)) * 100 if episode_rewards else 0
                
                # Food collection stats
                info['avg_food_per_episode'] = sum(episode_rewards) / len(episode_rewards) / 50 if episode_rewards else 0  # Rough estimate
            else:
                # Default values for first episode
                info['avg_reward'] = 0
                info['best_reward'] = 0
                info['recent_5_avg'] = 0
                info['trend'] = "🔄"
                info['trend_status'] = "STARTING"
                info['learning_status'] = "INITIALIZING"
                info['success_rate'] = 0
        elif self.ai:
            info['mode'] = 'Rule-based AI'
        else:
            info['mode'] = 'Manual'
        # Controls/help
        info['controls'] = [
            'Arrow keys: Move',
            'T: Toggle AI',
            'L: Toggle Learning AI',
            'M: Toggle Manual Teaching Mode',
            'P: Toggle Learning Pause',
            'S: Save Model',
            'Q: Quit Game',
            '+/-: Speed'
        ]
        # Only call GameRenderer methods/attributes if renderer is GameRenderer
        from ai_snake.render.renderer import GameRenderer
        if isinstance(self.renderer, GameRenderer):
            self.renderer.render(self.state_manager.game_state, current_time, info)
        else:
            self.renderer.render(self.state_manager.game_state, current_time, info)
    
    def reset(self):
        """Reset the game to initial state."""
        # Record episode end for learning AI
        if self.learning_ai:
            self.ai_manager.record_episode_end(self.state_manager.game_state.score, death_type=str(self.state_manager.game_state.death_type) if self.state_manager.game_state.death_type else "unknown")
            # Increment persistent episode ID
            self.episode_count += 1
            self._save_episode_id(self.episode_count)
            self.total_food_eaten += self.state_manager.game_state.score // 10  # Assuming 10 points per food
            if self.ai_manager.learning_ai_controller:
                self.ai_manager.learning_ai_controller.reward_calculator.reset()
            # Get the reward for the just-finished episode
            last_reward = 0
            if self.ai_manager.learning_ai_controller and self.ai_manager.learning_ai_controller.agent.episode_rewards:
                last_reward = self.ai_manager.learning_ai_controller.agent.episode_rewards[-1]
            death_type = str(self.state_manager.game_state.death_type) if self.state_manager.game_state.death_type else "other"
            # Add to both leaderboards using the persistent episode_count
            high_score_flag = self.state_manager.game_state.score == self.high_score
            self.leaderboard_service.add_entry(self.episode_count, last_reward, death_type, high_score=high_score_flag)
            # Auto-save based on config settings
            config = load_config('config/config.yaml')
            auto_save_enabled = config['learning'].get('auto_save_enabled', True)
            auto_save_interval = config['learning'].get('auto_save_interval', 1)  # Default to every episode
            auto_save_filename = config['learning'].get('auto_save_filename', 'snake_dqn_model_auto.pth')
            if auto_save_enabled and self.episode_count % auto_save_interval == 0:
                self.ai_manager.save_model(auto_save_filename)
        # Reset step counter for new episode
        self.step_count = 0
        self.state_manager.game_state.reset()
        self.death_type = None  # Reset death_type for new episode
    
    def run_game_loop(self) -> bool:
        """Run the main game loop. Returns True if game should continue, False to quit."""
        if self.headless:
            # Guarantee logging prints to console in headless mode
            root_logger = logging.getLogger()
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(logging.Formatter('[HEADLESS] %(message)s'))
            root_logger.addHandler(console_handler)
            root_logger.setLevel(logging.INFO)
            # TEST LOG: Confirm headless logger is active
            logger.info('Headless mode logger active. Only [HEADLESS] stats will be shown.')
            # Headless mode: no rendering, no event handling, run as fast as possible
            while not self.state_manager.game_state.game_over:
                self.update()
            # Game over - restart immediately
            if self.state_manager.game_state.score > self.high_score:
                self.save_high_score()
            # Remove print_performance_report call (does not exist)
            # --- HEADLESS MODE LOGGING OUTPUT ---
            if self.learning_ai and self.ai_manager.learning_ai_controller is not None:
                episode_count = self.episode_count
                episode_rewards = self.ai_manager.learning_ai_controller.agent.episode_rewards
                if episode_rewards:
                    current_reward = episode_rewards[-1]
                    avg_reward = sum(episode_rewards) / len(episode_rewards)
                    top_reward = max(episode_rewards)
                else:
                    current_reward = 0
                    avg_reward = 0
                    top_reward = 0
                highlight = ''
                if current_reward == top_reward and len(episode_rewards) > 1:
                    highlight = ' *** NEW HIGH SCORE! ***'
                logger.info(f"Episode: {episode_count} | Current reward: {current_reward:.2f} | Avg reward: {avg_reward:.2f} | High: {top_reward:.2f}{highlight}")
            self.reset()
            return self.run_game_loop()
        else:
            while not self.state_manager.game_state.game_over:
                for event in pygame.event.get():
                    if not self.input_handler.handle_input(event):
                        if self.learning_ai:
                            self.print_learning_report()
                        return False
                self.update()
                self.render()
                if not self.headless and self.clock is not None:
                    self.clock.tick(self.speed)
        # Game over - restart immediately
        if self.state_manager.game_state.score > self.high_score:
            self.save_high_score()
        # Auto-restart (no game over screen)
        self.reset()
        return self.run_game_loop()
    
    def print_learning_report(self):
        """Print a comprehensive learning analysis report."""
        if not self.ai_manager.learning_ai_controller:
            return
        
        stats = self.ai_manager.get_stats()
        episode_rewards = self.ai_manager.learning_ai_controller.agent.episode_rewards
        
        print("\n" + "="*60)
        print("🤖 AI LEARNING REPORT")
        print("="*60)
        
        # Basic stats
        print(f"📊 Total Episodes: {self.episode_count}")
        print(f"🧠 Training Steps: {stats.get('training_step', 0)}")
        print(f"💾 Memory Size: {stats.get('memory_size', 0)}")
        print(f"🎯 Current Epsilon: {stats.get('epsilon', 1.0):.3f}")
        
        if episode_rewards:
            # Performance analysis
            recent_rewards = episode_rewards[-20:] if len(episode_rewards) >= 20 else episode_rewards
            early_rewards = episode_rewards[:20] if len(episode_rewards) >= 20 else episode_rewards
            
            print(f"\n📈 PERFORMANCE ANALYSIS:")
            print(f"   Best Reward: {max(episode_rewards):.2f}")
            print(f"   Worst Reward: {min(episode_rewards):.2f}")
            print(f"   Average Reward: {sum(episode_rewards) / len(episode_rewards):.2f}")
            
            if len(episode_rewards) >= 20:
                recent_avg = sum(recent_rewards) / len(recent_rewards)
                early_avg = sum(early_rewards) / len(early_rewards)
                improvement = recent_avg - early_avg
                
                print(f"   Recent 20 Avg: {recent_avg:.2f}")
                print(f"   Early 20 Avg: {early_avg:.2f}")
                print(f"   Overall Improvement: {improvement:+.2f}")
                
                # Learning assessment
                print(f"\n🎓 LEARNING ASSESSMENT:")
                if improvement > 2.0:
                    print("   ✅ EXCELLENT: AI is learning well!")
                elif improvement > 0.5:
                    print("   ✅ GOOD: AI is improving steadily")
                elif improvement > -0.5:
                    print("   ⚠️  STABLE: AI is maintaining performance")
                else:
                    print("   ❌ POOR: AI is not improving")
                
                # Trend analysis
                if len(episode_rewards) >= 10:
                    last_10 = episode_rewards[-10:]
                    last_10_avg = sum(last_10) / len(last_10)
                    short_term_improvement = last_10_avg - recent_avg
                    
                    print(f"   Recent 10 Avg: {last_10_avg:.2f}")
                    print(f"   Short-term Trend: {short_term_improvement:+.2f}")
                    
                    if short_term_improvement > 1.0:
                        print("   📈 ACCELERATING: Learning is speeding up!")
                    elif short_term_improvement > 0:
                        print("   📈 IMPROVING: Recent performance is better")
                    elif short_term_improvement > -1.0:
                        print("   ➡️  STABLE: Recent performance is steady")
                    else:
                        print("   📉 DECLINING: Recent performance is worse")
            
            # Exploration analysis
            epsilon = stats.get('epsilon', 1.0)
            print(f"\n🔍 EXPLORATION ANALYSIS:")
            print(f"   Current Exploration Rate: {epsilon:.1%}")
            if epsilon > 0.8:
                print("   🎲 HIGH EXPLORATION: AI is still very random")
            elif epsilon > 0.5:
                print("   🎲 MEDIUM EXPLORATION: AI is balancing learning and exploration")
            elif epsilon > 0.2:
                print("   🎯 LOW EXPLORATION: AI is mostly using learned knowledge")
            else:
                print("   🎯 VERY LOW EXPLORATION: AI is very confident in its knowledge")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if self.episode_count < 50:
            print("   🚀 Continue training - AI needs more episodes to learn")
        elif stats.get('epsilon', 1.0) > 0.7:
            print("   ⚡ Increase learning rate or adjust rewards - AI is still very random")
        elif not episode_rewards or max(episode_rewards) < 5:
            print("   🎯 Adjust reward system - AI is not achieving good scores")
        else:
            print("   ✅ AI is learning well - consider saving the model")
        
        print("="*60)
        print("Report generated on quit. Press Q to exit.")
        print("="*60 + "\n")
        
        # Print detailed game analysis if learning AI was used
        if self.learning_ai:
            try:
                if log_file_path and os.path.exists(log_file_path):
                    print_game_analysis(log_file_path)
            except Exception as e:
                print(f"Could not analyze game log: {e}")
    
    def get_settings(self) -> dict:
        """Get current game settings."""
        from ai_snake.render.renderer import GameRenderer
        settings = {
            'speed': self.speed,
            'grid': (self.state_manager.game_state.grid_width, self.state_manager.game_state.grid_height),
        }
        if isinstance(self.renderer, GameRenderer):
            settings['nes'] = self.renderer.nes_mode
        else:
            settings['nes'] = False
        return settings
    
    def update_settings(self, settings: dict):
        """Update game settings."""
        self.speed = settings.get('speed', self.speed)
        grid = settings.get('grid', (self.state_manager.game_state.grid_width, self.state_manager.game_state.grid_height))
        self.state_manager.game_state.grid_width, self.state_manager.game_state.grid_height = grid
        from ai_snake.render.renderer import GameRenderer
        if isinstance(self.renderer, GameRenderer):
            self.renderer.set_grid_size(grid[0], grid[1])
            self.renderer.nes_mode = settings.get('nes', self.renderer.nes_mode)
            # Only set font if nes_mode is True
            if self.renderer.nes_mode:
                self.renderer.font = pygame.font.SysFont('Courier', 16, bold=True)
            else:
                self.renderer.font = pygame.font.SysFont('Arial', 24)
    
    def save_model(self, filepath: str = "snake_dqn_model.pth"):
        """Save the learning model."""
        if self.ai_manager.learning_ai_controller:
            self.ai_manager.save_model(filepath)
    
    def load_model(self, filepath: str = "snake_dqn_model.pth"):
        """Load a learning model."""
        if self.ai_manager.learning_ai_controller:
            self.ai_manager.load_model(filepath) 