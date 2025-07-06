import pygame
import os
from typing import Optional, Tuple
from .models import GameState
from .renderer import GameRenderer
from .ai_controller import AIController
from .config import HIGH_SCORE_FILE


class GameController:
    """Manages the game loop and coordinates between game state, AI, and rendering."""
    
    def __init__(self, speed: int = 10, ai: bool = False, grid: Tuple[int, int] = (20, 20), 
                 nes_mode: bool = False):
        self.game_state = GameState(grid_width=grid[0], grid_height=grid[1])
        self.renderer = GameRenderer(nes_mode=nes_mode)
        self.ai_controller = AIController()
        self.speed = speed
        self.ai = ai
        self.clock = pygame.time.Clock()
        self.high_score = self.load_high_score()
        self.game_state.high_score = self.high_score
    
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
            elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                self.speed += 1
            elif event.key == pygame.K_MINUS:
                self.speed = max(1, self.speed - 1)
            elif not self.ai:
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
        
        self.game_state.move_snake()
        current_time = pygame.time.get_ticks()
        self.game_state.check_collision(current_time)
        self.game_state.handle_growth()
        
        # Update high score if needed
        if self.game_state.score > self.high_score:
            self.high_score = self.game_state.score
            self.game_state.high_score = self.high_score
    
    def render(self):
        """Render the current game state."""
        current_time = pygame.time.get_ticks()
        self.renderer.render_game(self.game_state, current_time)
    
    def reset(self):
        """Reset the game to initial state."""
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
        
        self.renderer.draw_game_over_screen(self.game_state.score, self.high_score)
        
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