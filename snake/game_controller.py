import pygame
import os
from typing import Optional, Tuple
from .models import GameState
from .renderer import GameRenderer
from .ai_controller import AIController
from .learning_ai_controller import LearningAIController, RewardCalculator
from .config import HIGH_SCORE_FILE


class GameController:
    """Manages the game loop and coordinates between game state, AI, and rendering."""
    
    def __init__(self, speed: int = 10, ai: bool = False, grid: Tuple[int, int] = (20, 20), 
                 nes_mode: bool = False, ai_tracing: bool = False, auto_advance: bool = False,
                 learning_ai: bool = False, model_path: Optional[str] = None):
        self.game_state = GameState(grid_width=grid[0], grid_height=grid[1])
        self.renderer = GameRenderer(nes_mode=nes_mode)
        self.ai_controller = AIController(enable_tracing=ai_tracing)
        self.learning_ai_controller = None
        self.reward_calculator = None
        
        if learning_ai:
            self.learning_ai_controller = LearningAIController(
                grid_size=(grid[0], grid[1]), 
                model_path=model_path,
                training=True
            )
            self.reward_calculator = RewardCalculator()
        
        self.speed = speed
        self.ai = ai
        self.learning_ai = learning_ai
        self.ai_tracing = ai_tracing
        self.auto_advance = auto_advance
        self.clock = pygame.time.Clock()
        self.high_score = self.load_high_score()
        self.game_state.high_score = self.high_score
        
        # Training stats
        self.episode_count = 0
        self.total_food_eaten = 0
    
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
            return False
        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return False  # Signal to pause
            elif event.key == pygame.K_t:
                self.ai = not self.ai
            elif event.key == pygame.K_l:
                # Toggle learning AI
                if self.learning_ai_controller:
                    self.learning_ai = not self.learning_ai
                    print(f"Learning AI: {'ON' if self.learning_ai else 'OFF'}")
            elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                self.speed += 1
            elif event.key == pygame.K_MINUS:
                self.speed = max(1, self.speed - 1)
            elif not self.ai and not self.learning_ai:
                self.handle_direction_input(event.key)
        
        return True
    
    def handle_direction_input(self, key: int):
        """Handle direction input for manual control."""
        if key == pygame.K_UP:
            self.game_state.set_direction((0, -1))
        elif key == pygame.K_DOWN:
            self.game_state.set_direction((0, 1))
        elif key == pygame.K_LEFT:
            self.game_state.set_direction((-1, 0))
        elif key == pygame.K_RIGHT:
            self.game_state.set_direction((1, 0))
    
    def update(self):
        """Update game state for one frame."""
        if self.ai:
            self.ai_controller.make_move(self.game_state)
        elif self.learning_ai and self.learning_ai_controller:
            # Get action from learning AI
            direction = self.learning_ai_controller.get_action(self.game_state)
            self.game_state.set_direction(direction)
        
        # Store state before move for learning
        if self.learning_ai and self.learning_ai_controller:
            old_score = self.game_state.score
        
        self.game_state.move_snake()
        current_time = pygame.time.get_ticks()
        
        self.game_state.check_collision(current_time)
        
        # Record AI events after collision detection
        if self.ai:
            self.ai_controller.check_food_eaten(self.game_state)
        
        self.game_state.handle_growth()
        
        # Record learning step
        if self.learning_ai and self.learning_ai_controller and self.reward_calculator:
            reward = self.reward_calculator.calculate_reward(self.game_state, self.game_state.game_over)
            self.learning_ai_controller.record_step(self.game_state, reward, self.game_state.game_over)
        
        # Update high score if needed
        if self.game_state.score > self.high_score:
            self.high_score = self.game_state.score
            self.game_state.high_score = self.high_score
    
    def render(self):
        """Render the current game state."""
        current_time = pygame.time.get_ticks()
        
        # Get learning stats if available
        learning_stats = None
        if self.learning_ai and self.learning_ai_controller:
            stats = self.learning_ai_controller.get_stats()
            if stats:
                # Calculate additional stats
                episode_rewards = self.learning_ai_controller.agent.episode_rewards
                if len(episode_rewards) >= 20:
                    recent_rewards = episode_rewards[-20:]
                    early_rewards = episode_rewards[:20]
                    stats['recent_avg'] = sum(recent_rewards) / len(recent_rewards)
                    stats['improvement'] = stats['recent_avg'] - (sum(early_rewards) / len(early_rewards))
                else:
                    stats['recent_avg'] = 0
                    stats['improvement'] = 0
                learning_stats = stats
        
        self.renderer.render_game(self.game_state, current_time, learning_stats, self.episode_count)
    
    def reset(self):
        """Reset the game to initial state."""
        # Record episode end for learning AI
        if self.learning_ai and self.learning_ai_controller:
            self.learning_ai_controller.record_episode_end(self.game_state.score)
            self.episode_count += 1
            self.total_food_eaten += self.game_state.score // 10  # Assuming 10 points per food
            if self.reward_calculator:
                self.reward_calculator.reset()
            
            # Print episode stats every 100 episodes
            if self.episode_count % 100 == 0:
                stats = self.learning_ai_controller.get_stats()
                print(f"Episode {self.episode_count}: Avg Reward={stats.get('avg_reward', 0):.2f}, "
                      f"Epsilon={stats.get('epsilon', 0):.3f}, Best={stats.get('best_reward', 0):.2f}")
        
        self.game_state.reset()
    
    def run_game_loop(self) -> bool:
        """Run the main game loop. Returns True if game should continue, False to quit."""
        while not self.game_state.game_over:
            for event in pygame.event.get():
                if not self.handle_input(event):
                    return False
            
            self.update()
            self.render()
            self.clock.tick(self.speed)
        
        # Game over
        if self.game_state.score > self.high_score:
            self.save_high_score()
        
        # Print AI performance report if tracing is enabled
        if self.ai and self.ai_tracing:
            self.ai_controller.print_performance_report()
        
        self.renderer.draw_game_over_screen(self.game_state.score, self.high_score)
        
        if self.auto_advance:
            # Immediately return to allow next game to start
            return True
        
        # Wait for user input
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        waiting = False
                    elif event.key == pygame.K_r:
                        self.reset()
                        return self.run_game_loop()
                    elif event.key == pygame.K_s and self.learning_ai_controller:
                        # Save model
                        self.learning_ai_controller.save_model("snake_dqn_model.pth")
                        print("Model saved!")
            self.clock.tick(5)
        
        return True
    
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