# FIXME: Review this file for potential issues or improvements
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
from ai_snake.render.base import BaseRenderer
from ai_snake.ai.rule_based import AIController
from ai_snake.ai.learning import LearningAIController, RewardCalculator, get_log_file_path
from ai_snake.config.config import HIGH_SCORE_FILE
from ai_snake.config.loader import load_config, CONFIG_FILE
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
                 learning_ai: bool = False, model_path: Optional[str] = None, headless: bool = False, web: bool = False, starvation_threshold: Optional[int] = None, log_to_file: bool = False):
        self.headless = headless
        self.web = web
        if self.headless:
            self.renderer = HeadlessRenderer()
            self.clock = None
        elif self.web:
            # Lazy import so Flask is only required for web mode.
            from ai_snake.render.web import WebRenderer
            self.renderer = WebRenderer()
            self.clock = None
        else:
            self.renderer = GameRenderer(nes_mode=nes_mode, headless=headless)
            self.clock = pygame.time.Clock()
        self.state_manager = GameStateManager(grid[0], grid[1])
        self.ai_manager = AIManager(grid_size=grid, ai_tracing=ai_tracing, learning_ai=learning_ai, model_path=model_path, log_to_file=log_to_file)
        self.model_path = model_path  # Store model_path for info area
        self.starvation_threshold = starvation_threshold if starvation_threshold is not None else 50
        
        # Load config for reward system
        config = load_config(CONFIG_FILE)
        
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
        
        self.input_handler = InputHandler(self)
        self.leaderboard_service = LeaderboardService()
        self.manual_teaching_mode = False

    @property
    def game_state(self):
        """Shortcut to the underlying GameState — avoids deep chaining."""
        return self.state_manager.game_state
    
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
    
    def _apply_ai_move(self):
        """Let the active AI (rule-based or learning) choose a direction."""
        if self.ai:
            self.ai_manager.make_move(self.game_state)
        elif self.learning_ai:
            if getattr(self, 'manual_teaching_mode', False):
                return  # manual input overrides AI
            direction = self.ai_manager.get_action(self.game_state)
            if direction is not None:
                self.game_state.set_direction(direction, force=True)

    def _apply_learning_step(self):
        """Calculate reward, enforce starvation, and record a learning step."""
        if not (self.learning_ai and self.ai_manager.learning_ai_controller):
            return
        lc = self.ai_manager.learning_ai_controller
        gs = self.game_state
        reward = lc.reward_calculator.calculate_reward(gs, gs.game_over)
        # Starvation death logic
        if lc.reward_calculator.moves_without_food >= lc.reward_calculator.starvation_threshold:
            gs.set_starvation_death()
        self.ai_manager.record_step(gs, reward, gs.game_over)
        self.current_reward = reward
        self.step_count = getattr(self, 'step_count', 0) + 1

    def _advance_game(self, current_time: int = 0):
        """Move the snake, detect collisions, and update high score."""
        gs = self.game_state
        self.state_manager.move_snake()
        gs.check_collision(current_time)
        # Set death type for wall/self collision if not already set
        if gs.game_over and gs.death_type is None:
            gs.set_other_death()
        # Update high score
        if gs.score > self.high_score:
            self.high_score = gs.score
            gs.high_score = self.high_score

    def update(self):
        """Update game state for one frame."""
        self._apply_ai_move()

        if self.headless:
            self._advance_game(current_time=0)
            self._apply_learning_step()
        else:
            if self.ai:
                self.ai_manager.check_food_eaten(self.game_state)
            current_time = pygame.time.get_ticks()
            self._advance_game(current_time=current_time)
            self._apply_learning_step()

        
    def _build_learning_stats(self, info: dict) -> None:
        """Populate *info* with learning AI statistics for the stats panel."""
        stats = self.ai_manager.get_stats()
        episode_rewards = []
        if self.ai_manager.learning_ai_controller is not None:
            episode_rewards = self.ai_manager.learning_ai_controller.agent.episode_rewards

        # Basic stats (always shown)
        info['deaths'] = self.episode_count
        info['episode'] = self.episode_count
        info['epsilon'] = stats.get('epsilon', 1.0)
        info['memory_size'] = stats.get('memory_size', 0)
        info['training_step'] = stats.get('training_step', 0)
        info['exploration_rate'] = f"{stats.get('epsilon', 1.0):.1%}"

        # Episode progress indicator
        if episode_rewards:
            info['episode_progress'] = f"Episode {self.episode_count + 1}"
            info['total_episodes_completed'] = len(episode_rewards)
            info['episode_percentage'] = 100.0
        else:
            info['episode_progress'] = "Episode 1 (Starting)"
            info['total_episodes_completed'] = 0
            info['episode_percentage'] = 0

        # Real-time learning info
        info['current_reward'] = getattr(self, 'current_reward', 0)
        info['teaching_mode'] = getattr(self, 'manual_teaching_mode', False)
        if self.ai_manager.learning_ai_controller is not None:
            info['learning_paused'] = not self.ai_manager.learning_ai_controller.training
        else:
            info['learning_paused'] = False

        if not episode_rewards:
            info.update({
                'avg_reward': 0, 'best_reward': 0, 'recent_5_avg': 0,
                'trend': "🔄", 'trend_status': "STARTING",
                'learning_status': "INITIALIZING", 'success_rate': 0,
            })
            return

        # Performance stats
        info['avg_reward'] = stats.get('avg_reward', 0)
        info['best_reward'] = stats.get('best_reward', 0)

        n = len(episode_rewards)

        # Trend analysis (requires ≥5 episodes)
        if n >= 5:
            recent_5 = episode_rewards[-5:]
            recent_5_avg = sum(recent_5) / 5
            info['recent_5_avg'] = recent_5_avg

            if n >= 10:
                previous_5_avg = sum(episode_rewards[-10:-5]) / 5
                info['previous_5_avg'] = previous_5_avg
                info['short_term_trend'] = recent_5_avg - previous_5_avg

            if n >= 20:
                info['recent_20_avg'] = sum(episode_rewards[-20:]) / 20
                info['early_20_avg'] = sum(episode_rewards[:20]) / 20
                info['long_term_improvement'] = info['recent_20_avg'] - info['early_20_avg']

        if n >= 3:
            info['last_3_avg'] = sum(episode_rewards[-3:]) / 3
            info['last_episode'] = episode_rewards[-1]
            info['second_last_episode'] = episode_rewards[-2] if n >= 2 else 0
            info['third_last_episode'] = episode_rewards[-3] if n >= 3 else 0
            info['trend'] = "📈" if episode_rewards[-1] > episode_rewards[-3] else "📉"

            if n >= 5 and 'previous_5_avg' in info:
                if info['recent_5_avg'] > info['previous_5_avg']:
                    info['trend_status'], info['trend_color'] = "IMPROVING", "green"
                elif info['recent_5_avg'] < info['previous_5_avg']:
                    info['trend_status'], info['trend_color'] = "DECLINING", "red"
                else:
                    info['trend_status'], info['trend_color'] = "STABLE", "yellow"

        # Learning speed
        if n >= 10:
            learning_speed = sum(episode_rewards[-10:]) / 10 - sum(episode_rewards[:10]) / 10
            info['learning_speed'] = learning_speed
            if learning_speed > 5:
                info['learning_status'] = "FAST"
            elif learning_speed > 0:
                info['learning_status'] = "STEADY"
            else:
                info['learning_status'] = "SLOW"

        info['total_episodes'] = n
        info['current_episode'] = self.episode_count

        positive_episodes = sum(1 for r in episode_rewards if r > 0)
        info['success_rate'] = (positive_episodes / n) * 100
        info['avg_food_per_episode'] = sum(episode_rewards) / n / 50  # Rough estimate

    def _build_render_info(self) -> dict:
        """Build the info dict consumed by the renderer."""
        info = {
            'score': self.game_state.score,
            'high_score': self.high_score,
            'speed': self.speed,
        }
        if self.learning_ai:
            info['mode'] = 'Learning AI'
            if self.model_path:
                info['model'] = self.model_path
            self._build_learning_stats(info)
        elif self.ai:
            info['mode'] = 'Rule-based AI'
        else:
            info['mode'] = 'Manual'

        info['controls'] = [
            'Arrow keys: Move', 'T: Toggle AI', 'L: Toggle Learning AI',
            'M: Toggle Manual Teaching Mode', 'P: Toggle Learning Pause',
            'S: Save Model', 'Q: Quit Game', '+/-: Speed',
        ]
        return info

    def render(self):
        """Render the current game state."""
        if self.headless:
            return
        current_time = pygame.time.get_ticks()
        info = self._build_render_info()
        # Only call GameRenderer methods/attributes if renderer is GameRenderer
        from ai_snake.render.renderer import GameRenderer
        if isinstance(self.renderer, GameRenderer):
            self.renderer.render(self.game_state, current_time, info)
        else:
            self.renderer.render(self.game_state, current_time, info)
    
    def reset(self):
        """Reset the game to initial state."""
        # Log death for rule-based AI before reset
        if self.ai and hasattr(self.ai_manager, 'ai_controller'):
            death_type = str(self.game_state.death_type) if self.game_state.death_type else "unknown"
            move_count = getattr(self.ai_manager.ai_controller, 'move_count', 0)
            self.ai_manager.ai_controller.log_death(death_type, move_count)
            self.ai_manager.ai_controller.reset_stats()
        # Record episode end for learning AI
        if self.learning_ai:
            self.ai_manager.record_episode_end(self.game_state.score, death_type=str(self.game_state.death_type) if self.game_state.death_type else "unknown")
            # Increment persistent episode ID
            self.episode_count += 1
            self._save_episode_id(self.episode_count)
            self.total_food_eaten += self.game_state.score // 10  # Assuming 10 points per food
            if self.ai_manager.learning_ai_controller:
                self.ai_manager.learning_ai_controller.reward_calculator.reset()
            # Get the reward for the just-finished episode
            last_reward = 0
            if self.ai_manager.learning_ai_controller and self.ai_manager.learning_ai_controller.agent.episode_rewards:
                last_reward = self.ai_manager.learning_ai_controller.agent.episode_rewards[-1]
            death_type = str(self.game_state.death_type) if self.game_state.death_type else "other"
            # Add to both leaderboards using the persistent episode_count
            # Determine if this is a session high score by comparing reward with session's highest reward
            session_entries = self.leaderboard_service.get_session_entries()
            session_highest_reward = max([entry.get('reward', 0) for entry in session_entries]) if session_entries else 0
            high_score_flag = last_reward >= session_highest_reward and last_reward > 0
            self.leaderboard_service.add_entry(self.episode_count, last_reward, death_type, high_score=high_score_flag)
            # Auto-save based on config settings
            config = load_config(CONFIG_FILE)
            auto_save_enabled = config['learning'].get('auto_save_enabled', True)
            auto_save_interval = config['learning'].get('auto_save_interval', 1)  # Default to every episode
            auto_save_filename = config['learning'].get('auto_save_filename', 'snake_dqn_model_auto.pth')
            if auto_save_enabled and self.episode_count % auto_save_interval == 0:
                self.ai_manager.save_model(auto_save_filename)
        # Reset step counter for new episode
        self.step_count = 0
        self.game_state.reset()
        self.death_type = None  # Reset death_type for new episode
    
    def run_game_loop(self) -> bool:
        """Run the main game loop. Returns True if game should continue, False to quit."""
        if self.headless:
            # Guarantee logging prints to console in headless mode (configure once).
            root_logger = logging.getLogger()
            if not getattr(self, "_headless_logging_configured", False):
                for handler in root_logger.handlers[:]:
                    root_logger.removeHandler(handler)
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setFormatter(logging.Formatter('[HEADLESS] %(message)s'))
                root_logger.addHandler(console_handler)
                root_logger.setLevel(logging.INFO)
                self._headless_logging_configured = True
                logger.info('Headless mode logger active. Only [HEADLESS] stats will be shown.')

            # Headless mode: run episodes forever (caller can Ctrl+C to stop).
            while True:
                while not self.game_state.game_over:
                    self.update()

                if self.game_state.score > self.high_score:
                    self.save_high_score()

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
                    logger.info(
                        f"Episode: {episode_count} | Current reward: {current_reward:.2f} | "
                        f"Avg reward: {avg_reward:.2f} | High: {top_reward:.2f}{highlight}"
                    )

                self.reset()
        else:
            # Interactive mode
            running = True
            while running:
                # Main game loop
                while not self.game_state.game_over:
                    for event in pygame.event.get():
                        if not self.input_handler.handle_input(event):
                            if self.learning_ai:
                                self.print_learning_report()
                            return False
                    self.update()
                    self.render()
                    if not self.headless and self.clock is not None:
                        self.clock.tick(self.speed)
                
                # Game over handling
                if self.game_state.score > self.high_score:
                    self.save_high_score()
                
                # Auto-restart (no game over screen)
                self.reset()
                
            return False
    
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
        if self.ai_manager.learning_ai_controller:
            stats = self.ai_manager.learning_ai_controller.episode_stats
            if stats:
                print(f"\n💀 DEATH TYPE BREAKDOWN:")
                death_types = {}
                for ep in stats:
                    dt = ep.get('death_type', 'unknown')
                    death_types[dt] = death_types.get(dt, 0) + 1
                for dt, count in sorted(death_types.items(), key=lambda x: x[1], reverse=True):
                    print(f"   {dt.capitalize()}: {count} ({count/len(stats):.1%})")
    
    def get_settings(self) -> dict:
        """Get current game settings."""
        from ai_snake.render.renderer import GameRenderer
        settings = {
            'speed': self.speed,
            'grid': (self.game_state.grid_width, self.game_state.grid_height),
        }
        if isinstance(self.renderer, GameRenderer):
            settings['nes'] = self.renderer.nes_mode
        else:
            settings['nes'] = False
        return settings
    
    def update_settings(self, settings: dict):
        """Update game settings."""
        self.speed = settings.get('speed', self.speed)
        grid = settings.get('grid', (self.game_state.grid_width, self.game_state.grid_height))
        self.game_state.grid_width, self.game_state.grid_height = grid
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
