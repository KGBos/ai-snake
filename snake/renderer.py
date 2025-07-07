import pygame
from typing import Tuple, Optional
from .config import (
    BLACK, BLUE, GREEN, RED, WHITE, SCREEN_WIDTH, SCREEN_HEIGHT,
    FONT, FONT_SMALL, RETRO_FONT
)
from .models import GameState


class GameRenderer:
    """Handles all rendering logic for the snake game."""
    
    def __init__(self, nes_mode: bool = False):
        self.nes_mode = nes_mode
        self.font = RETRO_FONT if nes_mode else FONT
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('AI Snake')
        self.cell_size = None  # Will be set when grid size is known
        self.game_area_height = None  # Height of the game grid area
        self.stats_area_y = None  # Y position where stats start
    
    def set_grid_size(self, grid_width: int, grid_height: int):
        """Set the grid size and calculate cell size."""
        # Reserve space for stats area at the bottom
        stats_area_height = 120  # Height for learning statistics
        available_height = SCREEN_HEIGHT - stats_area_height
        
        self.cell_size = min(SCREEN_WIDTH // grid_width,
                            available_height // grid_height)
        
        # Calculate game area dimensions
        self.game_area_height = grid_height * self.cell_size
        self.stats_area_y = self.game_area_height
    
    def draw_cell(self, pos: Tuple[int, int], color: Tuple[int, int, int]):
        """Draw a single cell at the given position."""
        if self.cell_size is None:
            return
            
        x, y = pos
        rect = pygame.Rect(x * self.cell_size, y * self.cell_size,
                          self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, color, rect, border_radius=4)
        if self.nes_mode:
            pygame.draw.rect(self.screen, WHITE, rect, 1)
    
    def draw_grid(self, grid_width: int, grid_height: int):
        """Draw the grid lines in NES mode."""
        if not self.nes_mode or self.cell_size is None:
            return
            
        for x in range(grid_width):
            pygame.draw.line(self.screen, WHITE, 
                           (x * self.cell_size, 0), 
                           (x * self.cell_size, self.game_area_height), 1)
        for y in range(grid_height):
            pygame.draw.line(self.screen, WHITE, 
                           (0, y * self.cell_size), 
                           (SCREEN_WIDTH, y * self.cell_size), 1)
    
    def draw_snake(self, snake_body: list):
        """Draw the snake body."""
        for i, segment in enumerate(snake_body):
            color = BLUE if i == 0 else GREEN
            self.draw_cell(segment, color)
    
    def draw_food(self, food_pos: Tuple[int, int]):
        """Draw the food."""
        self.draw_cell(food_pos, RED)
    
    def draw_score(self, score: int, multiplier: int = 1):
        """Draw the score display in the game area."""
        score_text = self.font.render(f'Score: {score} x{multiplier}', True, WHITE)
        self.screen.blit(score_text, (5, 5))
    
    def draw_learning_stats(self, stats: dict, episode_count: int):
        """Draw learning statistics in the dedicated area below the game."""
        if self.stats_area_y is None:
            return
            
        # Clear the stats area
        stats_rect = pygame.Rect(0, self.stats_area_y, SCREEN_WIDTH, SCREEN_HEIGHT - self.stats_area_y)
        pygame.draw.rect(self.screen, BLACK, stats_rect)
        
        # Draw separator line
        if self.stats_area_y is not None:
            stats_y = int(self.stats_area_y)
            pygame.draw.line(self.screen, WHITE, (0, stats_y), (SCREEN_WIDTH, stats_y), 2)
        
        # Draw stats in columns
        y_offset = self.stats_area_y + 10
        line_height = 20
        
        # Left column
        left_x = 10
        stats_lines = [
            f"Episode: {episode_count}",
            f"Epsilon: {stats.get('epsilon', 0):.3f}",
            f"Training Steps: {stats.get('training_step', 0)}",
            f"Memory Size: {stats.get('memory_size', 0)}"
        ]
        
        for i, line in enumerate(stats_lines):
            text_surf = FONT_SMALL.render(line, True, WHITE)
            self.screen.blit(text_surf, (left_x, y_offset + i * line_height))
        
        # Right column
        right_x = SCREEN_WIDTH // 2 + 10
        reward_lines = [
            f"Avg Reward: {stats.get('avg_reward', 0):.2f}",
            f"Best Reward: {stats.get('best_reward', 0):.2f}",
            f"Recent Avg: {stats.get('recent_avg', 0):.2f}",
            f"Improvement: {stats.get('improvement', 0):+.2f}"
        ]
        
        for i, line in enumerate(reward_lines):
            text_surf = FONT_SMALL.render(line, True, WHITE)
            self.screen.blit(text_surf, (right_x, y_offset + i * line_height))
        
        # Bottom area for additional info
        bottom_y = self.stats_area_y + 90
        info_lines = [
            "Controls: L=Toggle AI, S=Save Model, ESC=Quit",
            "Training: Learning from experience, improving over time"
        ]
        
        for i, line in enumerate(info_lines):
            text_surf = FONT_SMALL.render(line, True, (200, 200, 200))
            self.screen.blit(text_surf, (10, bottom_y + i * 15))
    
    def draw_help_text(self, help_lines: list):
        """Draw help text in the stats area."""
        if self.stats_area_y is None:
            return
            
        for i, line in enumerate(help_lines):
            help_surf = FONT_SMALL.render(line, True, WHITE)
            self.screen.blit(help_surf, (5, self.stats_area_y + 60 + i * 15))
    
    def draw_game_over_screen(self, score: int, high_score: int):
        """Draw the game over screen."""
        self.screen.fill(BLACK)
        
        over_text = self.font.render('Game Over', True, WHITE)
        score_text = self.font.render(f'Score: {score}', True, WHITE)
        high_text = self.font.render(f'High: {high_score}', True, WHITE)
        prompt = self.font.render('Press Enter to return to Main Menu', True, WHITE)
        
        self.screen.blit(over_text, over_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 40)))
        self.screen.blit(score_text, score_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2)))
        self.screen.blit(high_text, high_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 40)))
        self.screen.blit(prompt, prompt.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 80)))
        
        pygame.display.flip()
    
    def render_game(self, game_state: GameState, current_time: int, learning_stats: Optional[dict] = None, episode_count: int = 0):
        """Render the complete game state."""
        self.screen.fill(BLACK)
        
        # Set grid size if not already set
        if self.cell_size is None:
            self.set_grid_size(game_state.grid_width, game_state.grid_height)
        
        # Draw game elements (only in the game area)
        self.draw_snake(game_state.get_snake_body())
        self.draw_food(game_state.food)
        self.draw_grid(game_state.grid_width, game_state.grid_height)
        
        # Draw UI in game area
        multiplier = 2 if (game_state.last_food_time and 
                          current_time - game_state.last_food_time <= 3000) else 1
        self.draw_score(game_state.score, multiplier)
        
        # Draw learning stats in dedicated area
        if learning_stats:
            self.draw_learning_stats(learning_stats, episode_count)
        else:
            # Draw regular help text if no learning stats
            help_lines = [
                'Arrow keys: Move',
                'T: Toggle AI',
                'Esc: Pause',
                '+/-: Speed',
                'R: Restart'
            ]
            self.draw_help_text(help_lines)
        
        pygame.display.flip()
    
    def clear_screen(self):
        """Clear the screen."""
        self.screen.fill(BLACK)
        pygame.display.flip()


class MenuRenderer:
    """Handles rendering for menu screens."""
    
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    
    def draw_centered_text(self, text: str, y: int, color: Tuple[int, int, int] = WHITE):
        """Draw centered text at the specified y position."""
        surf = FONT.render(text, True, color)
        rect = surf.get_rect(center=(SCREEN_WIDTH // 2, y))
        self.screen.blit(surf, rect)
    
    def draw_menu_options(self, options: list, selection: int, start_y: int = 120, spacing: int = 40):
        """Draw menu options with selection highlighting."""
        for i, opt in enumerate(options):
            color = BLUE if selection == i else WHITE
            text = FONT.render(opt, True, color)
            if selection == i:
                text = FONT.render(opt, True, color, BLACK)
            rect = text.get_rect(center=(SCREEN_WIDTH // 2, start_y + i * spacing))
            self.screen.blit(text, rect)
    
    def render_main_menu(self, selection: int):
        """Render the main menu."""
        self.screen.fill(BLACK)
        self.draw_centered_text('AI Snake', 60)
        
        options = ['AI Game', 'Human Game', 'Settings', 'Quit']
        self.draw_menu_options(options, selection)
        
        pygame.display.flip()
    
    def render_settings_menu(self, settings: dict, selection: int):
        """Render the settings menu."""
        self.screen.fill(BLACK)
        self.draw_centered_text('Settings', 80)
        
        option_texts = [
            'Speed: Slow',
            'Speed: Normal',
            'Speed: Fast',
            'Grid: 10x10',
            'Grid: 20x20',
            'Grid: 30x30',
            f'NES Style: {"On" if settings.get("nes", False) else "Off"}',
            'Back'
        ]
        
        for i, val in enumerate(option_texts):
            color = BLUE if selection == i else WHITE
            text = FONT.render(val, True, color)
            rect = text.get_rect(center=(SCREEN_WIDTH // 2, 140 + i * 30))
            self.screen.blit(text, rect)
        
        pygame.display.flip()
    
    def render_pause_menu(self, selection: int):
        """Render the pause menu."""
        self.screen.fill(BLACK)
        self.draw_centered_text('Paused', 60)
        
        options = ['Resume', 'Toggle AI', 'Speed: Slow', 'Speed: Normal', 'Speed: Fast']
        self.draw_menu_options(options, selection)
        
        pygame.display.flip() 