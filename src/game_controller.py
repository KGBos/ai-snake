import os
import sys
# Set SDL_VIDEODRIVER to 'dummy' for true headless mode (no window)
if '--headless' in sys.argv or os.environ.get('SNAKE_HEADLESS') == '1':
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
import pygame
from typing import Optional, Tuple
from .models import GameState
from .renderer import GameRenderer
from .ai_controller import AIController
from .learning_ai_controller import LearningAIController, RewardCalculator, print_game_analysis
from .config import HIGH_SCORE_FILE
from .config_loader import load_config
import logging


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
                 learning_ai: bool = False, model_path: Optional[str] = None, headless: bool = False, starvation_threshold: Optional[int] = None):
        self.headless = headless
        if not self.headless:
            self.clock = pygame.time.Clock()
        else:
            self.clock = None
        self.game_state = GameState(grid_width=grid[0], grid_height=grid[1])
        self.renderer = GameRenderer(nes_mode=nes_mode, headless=headless)
        self.ai_controller = AIController(enable_tracing=ai_tracing)
        self.learning_ai_controller = None
        self.reward_calculator = None
        self.model_path = model_path  # Store model_path for info area
        self.starvation_threshold = starvation_threshold if starvation_threshold is not None else 50
        
        # Load config for reward system
        config = load_config('src/config.yaml')
        
        if learning_ai:
            self.learning_ai_controller = LearningAIController(
                grid_size=(grid[0], grid[1]), 
                model_path=model_path,
                training=True
            )
            self.reward_calculator = RewardCalculator(config, starvation_threshold=self.starvation_threshold)
        
        self.speed = speed
        self.ai = ai
        self.learning_ai = learning_ai
        self.ai_tracing = ai_tracing
        self.auto_advance = auto_advance
        self.high_score = self.load_high_score()
        self.game_state.high_score = self.high_score
        
        # Training stats
        self.episode_count = self._load_episode_id()  # Persistent episode ID
        self.total_food_eaten = 0
        
        # Debug mode for learning observation
        self.debug_learning = True  # Always show learning process
        
        self.death_type = None
    
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
    
    def handle_input(self, event: pygame.event.Event) -> bool:
        """Handle a single input event. Returns True if game should continue."""
        if event.type == pygame.QUIT:
            logging.info('Received QUIT event.')
            return False
        
        if event.type == pygame.KEYDOWN:
            logging.debug(f'Key pressed: {event.key}')
            if event.key == pygame.K_q:
                logging.info('Q pressed - quitting game.')
                return False  # Quit on Q key
            elif event.key == pygame.K_ESCAPE:
                logging.info('Pause/Quit triggered.')
                return False  # Signal to pause
            elif event.key == pygame.K_t:
                self.ai = not self.ai
                logging.info(f'Rule-based AI toggled: {self.ai}')
            elif event.key == pygame.K_l:
                # Toggle learning AI
                if self.learning_ai_controller:
                    self.learning_ai = not self.learning_ai
                    logging.info(f'Learning AI toggled: {self.learning_ai}')
            elif event.key == pygame.K_m:
                # Toggle manual teaching mode (override AI with manual input)
                if self.learning_ai and self.learning_ai_controller:
                    self.manual_teaching_mode = not getattr(self, 'manual_teaching_mode', False)
                    logging.info(f'Manual teaching mode: {self.manual_teaching_mode}')
            elif event.key == pygame.K_p:
                # Toggle learning pause (stop training but keep AI running)
                if self.learning_ai_controller:
                    self.learning_ai_controller.set_training_mode(not self.learning_ai_controller.training)
                    logging.info(f'Learning paused: {not self.learning_ai_controller.training}')
            elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                self.speed += 1
                logging.info(f'Increased speed to {self.speed}')
            elif event.key == pygame.K_MINUS:
                self.speed = max(1, self.speed - 1)
                logging.info(f'Decreased speed to {self.speed}')
            elif not self.ai and not self.learning_ai:
                self.handle_direction_input(event.key)
            elif self.learning_ai and getattr(self, 'manual_teaching_mode', False):
                # Manual input during teaching mode
                self.handle_direction_input(event.key)
        
        return True
    
    def handle_direction_input(self, key: int):
        """Handle direction input for manual control."""
        old_direction = self.game_state.direction
        if key == pygame.K_UP:
            self.game_state.set_direction((0, -1))
        elif key == pygame.K_DOWN:
            self.game_state.set_direction((0, 1))
        elif key == pygame.K_LEFT:
            self.game_state.set_direction((-1, 0))
        elif key == pygame.K_RIGHT:
            self.game_state.set_direction((1, 0))
        if self.game_state.direction != old_direction:
            logging.debug(f'Direction changed from {old_direction} to {self.game_state.direction}')
    
    def update(self):
        """Update game state for one frame."""
        if self.headless:
            # No rendering, no event handling
            if self.ai:
                self.ai_controller.make_move(self.game_state)
            elif self.learning_ai and self.learning_ai_controller:
                if getattr(self, 'manual_teaching_mode', False):
                    pass
                else:
                    direction = self.learning_ai_controller.get_action(self.game_state)
                    self.game_state.set_direction(direction, force=True)
            if self.learning_ai and self.learning_ai_controller and self.reward_calculator:
                reward = self.reward_calculator.calculate_reward(self.game_state, self.game_state.game_over)
                # STARVATION DEATH LOGIC
                if self.reward_calculator.moves_without_food >= self.reward_calculator.starvation_threshold:
                    self.game_state.game_over = True
                    self.game_state.death_type = 'starvation'
                self.learning_ai_controller.record_step(self.game_state, reward, self.game_state.game_over)
                self.current_reward = reward
                if hasattr(self, 'step_count'):
                    self.step_count += 1
                else:
                    self.step_count = 1
            old_head = self.game_state.get_snake_head()
            self.game_state.move_snake()
            new_head = self.game_state.get_snake_head()
            current_time = 0
            self.game_state.check_collision(current_time)
            # Set death type for wall/self collision
            if self.game_state.game_over and self.death_type is None:
                if not (0 <= new_head[0] < self.game_state.grid_width and 0 <= new_head[1] < self.game_state.grid_height):
                    self.death_type = 'wall'
                elif (new_head in list(self.game_state.snake)[1:]):
                    self.death_type = 'self'
                else:
                    self.death_type = 'other'
            if self.game_state.score > getattr(self, 'high_score', 0):
                self.high_score = self.game_state.score
                self.game_state.high_score = self.high_score
        else:
            if self.ai:
                logging.debug('Rule-based AI making move.')
                self.ai_controller.make_move(self.game_state)
            elif self.learning_ai and self.learning_ai_controller:
                # Check if in manual teaching mode
                if getattr(self, 'manual_teaching_mode', False):
                    # Manual input overrides AI in teaching mode
                    logging.debug('Manual teaching mode - AI disabled for manual input')
                else:
                    logging.debug('Learning AI making move.')
                    direction = self.learning_ai_controller.get_action(self.game_state)
                    self.game_state.set_direction(direction, force=True)
            
            # Store state before move for learning
            if self.learning_ai and self.learning_ai_controller:
                old_score = self.game_state.score
            
            old_head = self.game_state.get_snake_head()
            self.game_state.move_snake()
            new_head = self.game_state.get_snake_head()
            logging.debug(f'Snake moved from {old_head} to {new_head} in direction {self.game_state.direction}')
            current_time = pygame.time.get_ticks()
            
            self.game_state.check_collision(current_time)
            
            # Record AI events after collision detection
            if self.ai:
                self.ai_controller.check_food_eaten(self.game_state)
            
            # Record learning step (even during manual teaching)
            if self.learning_ai and self.learning_ai_controller and self.reward_calculator:
                reward = self.reward_calculator.calculate_reward(self.game_state, self.game_state.game_over)
                # STARVATION DEATH LOGIC
                if self.reward_calculator.moves_without_food >= self.reward_calculator.starvation_threshold:
                    self.game_state.game_over = True
                    self.game_state.death_type = 'starvation'
                self.learning_ai_controller.record_step(self.game_state, reward, self.game_state.game_over)
                # Store current reward for display
                self.current_reward = reward
                
                # Track step count for episode statistics
                if hasattr(self, 'step_count'):
                    self.step_count += 1
                else:
                    self.step_count = 1
                
                # Removed per-turn learning progress output - only log per episode now
            
            # Update high score if needed
            if self.game_state.score > self.high_score:
                self.high_score = self.game_state.score
                self.game_state.high_score = self.high_score
                logging.info(f'New high score: {self.high_score}')
            
            # Use death_type from GameState (set during collision detection)
            if self.game_state.game_over and self.game_state.death_type is None:
                self.game_state.death_type = 'other'  # Fallback for unknown causes
        
        # Add to leaderboard if learning AI
        if self.learning_ai and self.learning_ai_controller:
            # Get the reward for the just-finished episode
            if self.learning_ai_controller.agent.episode_rewards:
                last_reward = self.learning_ai_controller.agent.episode_rewards[-1]
            else:
                last_reward = 0
            death_type = str(self.game_state.death_type) if self.game_state.death_type else "other"
            # Determine if this episode set a new high score
            high_score_flag = self.game_state.score == self.high_score
            self.renderer.leaderboard.add_entry(self.episode_count, last_reward, death_type, high_score=high_score_flag)
    
    def render(self):
        """Render the current game state."""
        if self.headless:
            return
        current_time = pygame.time.get_ticks()
        # Build info dict for the info area
        info = {
            'score': self.game_state.score,
            'high_score': self.high_score,
            'speed': self.speed,
        }
        # Determine mode and model
        if self.learning_ai and self.learning_ai_controller:
            info['mode'] = 'Learning AI'
            if self.model_path:
                info['model'] = self.model_path
            # Add learning stats (always show, even before first death)
            stats = self.learning_ai_controller.get_stats()
            episode_rewards = self.learning_ai_controller.agent.episode_rewards
            
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
            info['learning_paused'] = not self.learning_ai_controller.training
            
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
        self.renderer.render_game(self.game_state, current_time, info)
    
    def reset(self):
        """Reset the game to initial state."""
        # Record episode end for learning AI
        if self.learning_ai and self.learning_ai_controller:
            self.learning_ai_controller.record_episode_end(self.game_state.score, death_type=str(self.game_state.death_type) if self.game_state.death_type else "unknown")
            # Increment persistent episode ID
            self.episode_count += 1
            self._save_episode_id(self.episode_count)
            self.total_food_eaten += self.game_state.score // 10  # Assuming 10 points per food
            if self.reward_calculator:
                self.reward_calculator.reset()
            # Get the reward for the just-finished episode
            if self.learning_ai_controller.agent.episode_rewards:
                last_reward = self.learning_ai_controller.agent.episode_rewards[-1]
            else:
                last_reward = 0
            death_type = str(self.game_state.death_type) if self.game_state.death_type else "other"
            # Add to both leaderboards using the persistent episode_count
            self.renderer.add_leaderboard_entry(self.episode_count, last_reward, death_type)
            # Auto-save based on config settings
            config = load_config('src/config.yaml')
            auto_save_enabled = config['learning'].get('auto_save_enabled', True)
            auto_save_interval = config['learning'].get('auto_save_interval', 1)  # Default to every episode
            auto_save_filename = config['learning'].get('auto_save_filename', 'snake_dqn_model_auto.pth')
            if auto_save_enabled and self.episode_count % auto_save_interval == 0:
                self.learning_ai_controller.save_model(auto_save_filename)
        # Reset step counter for new episode
        self.step_count = 0
        self.game_state.reset()
    
    def run_game_loop(self) -> bool:
        """Run the main game loop. Returns True if game should continue, False to quit."""
        if self.headless:
            # Headless mode: no rendering, no event handling, run as fast as possible
            while not self.game_state.game_over:
                self.update()
            # Game over - restart immediately
            if self.game_state.score > self.high_score:
                self.save_high_score()
            if self.ai and self.ai_tracing:
                self.ai_controller.print_performance_report()
            self.reset()
            return self.run_game_loop()
        else:
            while not self.game_state.game_over:
                for event in pygame.event.get():
                    if not self.handle_input(event):
                        if self.learning_ai and self.learning_ai_controller:
                            self.print_learning_report()
                        return False
                self.update()
                self.render()
                if not self.headless and self.clock is not None:
                    self.clock.tick(self.speed)
        
        # Game over - restart immediately
        if self.game_state.score > self.high_score:
            self.save_high_score()
        
        # Print AI performance report if tracing is enabled
        if self.ai and self.ai_tracing:
            self.ai_controller.print_performance_report()
        
        # Auto-restart (no game over screen)
        self.reset()
        return self.run_game_loop()
    
    def print_learning_report(self):
        """Print a comprehensive learning analysis report."""
        if not self.learning_ai_controller:
            return
        
        stats = self.learning_ai_controller.get_stats()
        episode_rewards = self.learning_ai_controller.agent.episode_rewards
        
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
        if self.learning_ai and self.learning_ai_controller:
            try:
                from .learning_ai_controller import log_file_path
                if log_file_path and os.path.exists(log_file_path):
                    print_game_analysis(log_file_path)
            except Exception as e:
                print(f"Could not analyze game log: {e}")
    
    def get_settings(self) -> dict:
        """Get current game settings."""
        return {
            'speed': self.speed,
            'grid': (self.game_state.grid_width, self.game_state.grid_height),
            'nes': self.renderer.nes_mode
        }
    
    def update_settings(self, settings: dict):
        """Update game settings."""
        self.speed = settings.get('speed', self.speed)
        grid = settings.get('grid', (self.game_state.grid_width, self.game_state.grid_height))
        self.game_state.grid_width, self.game_state.grid_height = grid
        self.renderer.set_grid_size(grid[0], grid[1])
        self.renderer.nes_mode = settings.get('nes', self.renderer.nes_mode)
        self.renderer.font = self.renderer.font = (self.renderer.font if not self.renderer.nes_mode 
                                                  else pygame.font.SysFont('Courier', 16, bold=True))
    
    def save_model(self, filepath: str = "snake_dqn_model.pth"):
        """Save the learning model."""
        if self.learning_ai_controller:
            self.learning_ai_controller.save_model(filepath)
    
    def load_model(self, filepath: str = "snake_dqn_model.pth"):
        """Load a learning model."""
        if self.learning_ai_controller:
            self.learning_ai_controller.load_model(filepath) 