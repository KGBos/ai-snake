import pygame
import logging
from typing import Tuple, Optional, List, Dict
from config.config import (
    BLACK, BLUE, GREEN, RED, WHITE, SCREEN_WIDTH, SCREEN_HEIGHT,
    FONT, FONT_SMALL, RETRO_FONT
)
from game.models import GameState
from config.loader import load_config, get_grid_padding, get_panel_padding, get_leaderboard_file
from render.leaderboard import Leaderboard
from render.base import BaseRenderer


class GameRenderer(BaseRenderer):
    """Handles all rendering logic for the snake game with three-panel layout."""
    
    def __init__(self, nes_mode: bool = False, headless: bool = False):
        self.nes_mode = nes_mode
        self.headless = headless
        self.font = RETRO_FONT if nes_mode else FONT
        if not self.headless:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption('AI Snake')
        else:
            self.screen = None
        
        # Load config
        config = load_config('src/config/config.yaml')
        self.panel_padding = get_panel_padding(config)
        self.grid_padding = get_grid_padding(config)
        self.leaderboard_file = get_leaderboard_file(config)
        
        # Initialize leaderboards
        self.leaderboard = Leaderboard(self.leaderboard_file)  # All-time leaderboard (persistent)
        self.session_leaderboard = Leaderboard(file_path=None)  # Session leaderboard (in-memory)
        
        # Layout variables
        self.left_panel_width = 0
        self.right_panel_width = 0
        self.game_area_width = 0
        self.game_area_height = 0
        self.cell_size = None
        self.grid_origin = (0, 0)
        
        # Calculate panel widths based on content
        self._calculate_panel_widths()
    
    def _calculate_panel_widths(self):
        """Calculate optimal panel widths based on content."""
        # Estimate left panel width (telemetry)
        telemetry_lines = [
            "Episodes: 999",
            "Episode: Episode 999 (Starting)",
            "Training Steps: 99999",
            "Memory: 99999",
            "Exploration: 99.9%",
            "Current Reward: 999.99",
            "Best Reward: 999.99",
            "Recent 5 Avg: 999.99",
            "Status: IMPROVING",
            "Learning: FAST",
            "Success Rate: 99.9%",
            "Avg Food: 99.99"
        ]
        
        max_telemetry_width = 0
        for line in telemetry_lines:
            text_surf = FONT_SMALL.render(line, True, WHITE)
            max_telemetry_width = max(max_telemetry_width, text_surf.get_width())
        
        # Estimate right panel width (leaderboard)
        leaderboard_lines = [
            "1. Episode 999 - 999 (WALL) ↑ +2",
            "2. Episode 999 - 999 (SELF) ↓ -1",
            "3. Episode 999 - 999 (STARVE) → NEW"
        ]
        
        max_leaderboard_width = 0
        for line in leaderboard_lines:
            text_surf = FONT_SMALL.render(line, True, WHITE)
            max_leaderboard_width = max(max_leaderboard_width, text_surf.get_width())
        
        # Add padding
        self.left_panel_width = max_telemetry_width + (self.panel_padding * 2)
        self.right_panel_width = max_leaderboard_width + (self.panel_padding * 2)
        
        # Calculate game area
        self.game_area_width = SCREEN_WIDTH - self.left_panel_width - self.right_panel_width
        self.game_area_height = SCREEN_HEIGHT
    
    def set_grid_size(self, grid_width: int, grid_height: int):
        """Set the grid size and calculate cell size for centered game area."""
        # Calculate cell size to fit in game area
        available_width = self.game_area_width - (self.grid_padding * 2)
        available_height = self.game_area_height - (self.grid_padding * 2)
        self.cell_size = min(available_width // grid_width, available_height // grid_height)
        
        # Calculate grid origin to center in game area
        grid_pixel_width = self.cell_size * grid_width
        grid_pixel_height = self.cell_size * grid_height
        
        # Center horizontally in game area
        game_area_x = self.left_panel_width
        origin_x = game_area_x + ((self.game_area_width - grid_pixel_width) // 2)
        
        # Center vertically
        origin_y = ((self.game_area_height - grid_pixel_height) // 2)
        
        self.grid_origin = (origin_x, origin_y)
        
        logging.info(f"Three-panel layout: Left={self.left_panel_width}, Game={self.game_area_width}, Right={self.right_panel_width}")
        logging.info(f"Grid: {grid_width}x{grid_height}, Cell size: {self.cell_size}, Origin: {self.grid_origin}")
    
    def draw_cell(self, pos: Tuple[int, int], color: Tuple[int, int, int]):
        """Draw a single cell at the given position."""
        if self.cell_size is None or self.headless or self.screen is None:
            return
        x, y = pos
        ox, oy = self.grid_origin
        rect = pygame.Rect(ox + x * self.cell_size, oy + y * self.cell_size,
                          self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, color, rect, border_radius=4)
        if self.nes_mode:
            pygame.draw.rect(self.screen, WHITE, rect, 1)
    
    def draw_grid(self, grid_width: int, grid_height: int):
        """Draw the grid lines in NES mode."""
        if self.headless or self.screen is None or not self.nes_mode or self.cell_size is None:
            return
        ox, oy = self.grid_origin
        for x in range(grid_width):
            pygame.draw.line(self.screen, WHITE, 
                           (ox + x * self.cell_size, oy), 
                           (ox + x * self.cell_size, oy + self.cell_size * grid_height), 1)
        for y in range(grid_height):
            pygame.draw.line(self.screen, WHITE, 
                           (ox, oy + y * self.cell_size), 
                           (ox + self.cell_size * grid_width, oy + y * self.cell_size), 1)
    
    def draw_snake(self, snake_body: list, learning_status: str = "normal"):
        """Draw the snake body with colored head based on learning status."""
        for i, segment in enumerate(snake_body):
            if i == 0:  # Snake head
                if learning_status == "teaching":
                    color = (255, 255, 0)  # Yellow for teaching mode
                elif learning_status == "paused":
                    color = (255, 100, 100)  # Red for learning paused
                else:
                    color = (100, 255, 100)  # Green for normal learning
            else:
                color = GREEN  # Body segments stay green
            self.draw_cell(segment, color)
    
    def draw_food(self, food_pos: Tuple[int, int]):
        """Draw the food."""
        self.draw_cell(food_pos, RED)
    
    def draw_left_panel(self, info: dict):
        """Draw the left telemetry panel."""
        if self.headless or self.screen is None:
            return
        
        # Clear left panel
        left_rect = pygame.Rect(0, 0, self.left_panel_width, SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, BLACK, left_rect)
        
        # Draw panel border
        pygame.draw.line(self.screen, WHITE, (self.left_panel_width, 0), 
                        (self.left_panel_width, SCREEN_HEIGHT), 2)
        
        y_offset = 10
        line_height = 18
        x_offset = self.panel_padding
        
        # Check if learning AI is active
        learning_active = info.get('mode') == 'Learning AI'
        
        # Basic game stats (always show)
        basic_stats = [
            f"Score: {info.get('score', 0)}",
            f"Speed: {info.get('speed', 0)}",
            f"Mode: {info.get('mode', 'Manual')}",
        ]
        if info.get('model'):
            basic_stats.append(f"Model: {info['model']}")
        
        for line in basic_stats:
            text_surf = FONT_SMALL.render(line, True, WHITE)
            self.screen.blit(text_surf, (x_offset, y_offset))
            y_offset += line_height
        
        y_offset += line_height // 2  # Extra spacing
        
        # Learning stats (if learning AI is active)
        if learning_active:
            # Status indicator
            status_text = "AI LEARNING"
            status_color = (100, 255, 100)  # Green
            
            if info.get('teaching_mode', False):
                status_text = "TEACHING MODE"
                status_color = (255, 255, 0)  # Yellow
            elif info.get('learning_paused', False):
                status_text = "LEARNING PAUSED"
                status_color = (255, 100, 100)  # Red
            
            # Draw status with background
            status_surf = FONT.render(status_text, True, status_color)
            status_rect = status_surf.get_rect()
            status_rect.topleft = (x_offset, y_offset)
            
            # Background for status
            bg_rect = pygame.Rect(x_offset - 5, y_offset - 5, 
                                 status_rect.width + 10, status_rect.height + 10)
            pygame.draw.rect(self.screen, (30, 30, 30), bg_rect)
            pygame.draw.rect(self.screen, status_color, bg_rect, 3)
            self.screen.blit(status_surf, status_rect)
            y_offset += status_rect.height + 15
            
            # Core learning stats
            core_stats = [
                f"Episodes: {info.get('deaths', 0)}",
                f"Episode: {info.get('episode_progress', 'Starting')}",
                f"Training Steps: {info.get('training_step', 0)}",
                f"Memory: {info.get('memory_size', 0)}",
                f"Exploration: {info.get('exploration_rate', '100%')}",
            ]
            
            for line in core_stats:
                text_surf = FONT_SMALL.render(line, True, WHITE)
                self.screen.blit(text_surf, (x_offset, y_offset))
                y_offset += line_height
            
            y_offset += line_height // 2
            
            # Performance stats
            perf_stats = [
                f"Current Reward: {info.get('current_reward', 0):.2f}",
                f"Best Reward: {info.get('best_reward', 0):.2f}",
                f"Recent 5 Avg: {info.get('recent_5_avg', 0):.2f}",
            ]
            
            if 'previous_5_avg' in info:
                perf_stats.append(f"Previous 5 Avg: {info['previous_5_avg']:.2f}")
            if 'short_term_trend' in info:
                perf_stats.append(f"Short-term: {info['short_term_trend']:+.2f}")
            if 'last_3_avg' in info:
                perf_stats.append(f"Last 3 Avg: {info['last_3_avg']:.2f}")
            
            for line in perf_stats:
                color = WHITE
                if "Short-term:" in line and info.get('short_term_trend', 0) > 0:
                    color = (100, 255, 100)  # Green for positive
                elif "Short-term:" in line and info.get('short_term_trend', 0) < 0:
                    color = (255, 100, 100)  # Red for negative
                
                text_surf = FONT_SMALL.render(line, True, color)
                self.screen.blit(text_surf, (x_offset, y_offset))
                y_offset += line_height
            
            y_offset += line_height // 2
            
            # Episode progress
            episode_progress = []
            if 'last_episode' in info:
                episode_progress.append(f"Last Episode: {info['last_episode']:.2f}")
            if 'second_last_episode' in info:
                episode_progress.append(f"2nd Last: {info['second_last_episode']:.2f}")
            if 'third_last_episode' in info:
                episode_progress.append(f"3rd Last: {info['third_last_episode']:.2f}")
            
            for line in episode_progress:
                color = WHITE
                if "Last Episode:" in line:
                    last_episode = info.get('last_episode', 0)
                    if last_episode > 0:
                        color = (100, 255, 100)  # Green for positive
                    elif last_episode < -10:
                        color = (255, 100, 100)  # Red for very negative
                    else:
                        color = (255, 165, 0)  # Orange for neutral
                
                text_surf = FONT_SMALL.render(line, True, color)
                self.screen.blit(text_surf, (x_offset, y_offset))
                y_offset += line_height
            
            y_offset += line_height // 2
            
            # Trend and learning status
            trend_stats = []
            if 'trend_status' in info:
                trend_stats.append(f"Status: {info['trend_status']}")
            if 'learning_status' in info:
                trend_stats.append(f"Learning: {info['learning_status']}")
            if 'success_rate' in info:
                trend_stats.append(f"Success Rate: {info['success_rate']:.1f}%")
            if 'avg_food_per_episode' in info:
                trend_stats.append(f"Avg Food: {info['avg_food_per_episode']:.2f}")
            
            for line in trend_stats:
                color = WHITE
                if "Status: IMPROVING" in line:
                    color = (100, 255, 100)  # Green
                elif "Status: DECLINING" in line:
                    color = (255, 100, 100)  # Red
                elif "Status: STABLE" in line:
                    color = (255, 255, 0)  # Yellow
                elif "Learning: FAST" in line:
                    color = (100, 255, 100)  # Green
                elif "Learning: STEADY" in line:
                    color = (255, 165, 0)  # Orange
                elif "Learning: SLOW" in line:
                    color = (255, 100, 100)  # Red
                elif "Success Rate:" in line:
                    success_rate = info.get('success_rate', 0)
                    if success_rate > 70:
                        color = (100, 255, 100)  # Green for high success
                    elif success_rate > 30:
                        color = (255, 165, 0)  # Orange for medium success
                    else:
                        color = (255, 100, 100)  # Red for low success
                
                text_surf = FONT_SMALL.render(line, True, color)
                self.screen.blit(text_surf, (x_offset, y_offset))
                y_offset += line_height
    
    def draw_right_panel(self):
        """Draw the right leaderboard panel."""
        if self.headless or self.screen is None:
            return
        
        # Clear right panel
        right_rect = pygame.Rect(SCREEN_WIDTH - self.right_panel_width, 0, 
                               self.right_panel_width, SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, BLACK, right_rect)
        
        # Draw panel border
        pygame.draw.line(self.screen, WHITE, (SCREEN_WIDTH - self.right_panel_width, 0), 
                        (SCREEN_WIDTH - self.right_panel_width, SCREEN_HEIGHT), 2)
        
        y_offset = 10
        line_height = 18
        x_offset = SCREEN_WIDTH - self.right_panel_width + self.panel_padding
        
        # --- SESSION LEADERBOARD ---
        session_title = FONT.render("SESSION TOP 10", True, (100, 255, 255))
        self.screen.blit(session_title, (x_offset, y_offset))
        y_offset += session_title.get_height() + 5
        session_entries = self.session_leaderboard.get_formatted_entries()
        if not session_entries:
            empty_surf = FONT_SMALL.render("No episodes yet", True, (100, 100, 100))
            self.screen.blit(empty_surf, (x_offset, y_offset))
            y_offset += line_height + 5
        else:
            for entry in session_entries:
                high_score_flag = " HighScore=true" if entry.get('high_score', False) else ""
                line = f"{entry['rank']}. Ep {entry['episode']} - Reward: {entry['reward']:.2f} ({entry['death_type']}){high_score_flag} {entry['arrow']} {entry['change_text']}"
                line_surf = FONT_SMALL.render(line, True, entry['color'])
                self.screen.blit(line_surf, (x_offset, y_offset))
                y_offset += line_height + 2
            # Show session stats
            stats = self.session_leaderboard.get_stats()
            if stats:
                y_offset += 5
                stats_text = f"Highest: {stats['highest_reward']:.2f}  Avg: {stats['average_reward']:.2f}  Lowest: {stats['lowest_reward']:.2f}"
                stats_surf = FONT_SMALL.render(stats_text, True, (200, 200, 200))
                self.screen.blit(stats_surf, (x_offset, y_offset))
                y_offset += line_height
        
        # --- ALL-TIME LEADERBOARD ---
        y_offset += 10
        alltime_title = FONT.render("ALL-TIME TOP 10", True, (255, 255, 100))
        self.screen.blit(alltime_title, (x_offset, y_offset))
        y_offset += alltime_title.get_height() + 5
        entries = self.leaderboard.get_formatted_entries()
        if not entries:
            empty_surf = FONT_SMALL.render("No episodes yet", True, (100, 100, 100))
            self.screen.blit(empty_surf, (x_offset, y_offset))
            return
        for entry in entries:
            high_score_flag = " HighScore=true" if entry.get('high_score', False) else ""
            line = f"{entry['rank']}. Ep {entry['episode']} - Reward: {entry['reward']:.2f} ({entry['death_type']}){high_score_flag} {entry['arrow']} {entry['change_text']}"
            line_surf = FONT_SMALL.render(line, True, entry['color'])
            self.screen.blit(line_surf, (x_offset, y_offset))
            y_offset += line_height + 2
        # Show leaderboard stats
        stats = self.leaderboard.get_stats()
        if stats:
            y_offset += 5
            stats_text = f"Highest: {stats['highest_reward']:.2f}  Avg: {stats['average_reward']:.2f}  Lowest: {stats['lowest_reward']:.2f}"
            stats_surf = FONT_SMALL.render(stats_text, True, (200, 200, 200))
            self.screen.blit(stats_surf, (x_offset, y_offset))
    
    def render(self, game_state: GameState, current_time: int, info: Optional[dict] = None):
        """Render the complete game state with three-panel layout."""
        if self.headless or self.screen is None:
            return
        
        self.screen.fill(BLACK)
        
        if self.cell_size is None:
            self.set_grid_size(game_state.grid_width, game_state.grid_height)
        
        # Determine learning status from info
        learning_status = "normal"
        if info and 'deaths' in info:  # Learning AI is active
            if info.get('teaching_mode', False):
                learning_status = "teaching"
            elif info.get('learning_paused', False):
                learning_status = "paused"
        
        # Draw game elements
        self.draw_snake(game_state.get_snake_body(), learning_status)
        self.draw_food(game_state.food)
        self.draw_grid(game_state.grid_width, game_state.grid_height)
        
        # Draw grid boundary overlay
        if self.cell_size is not None:
            ox, oy = self.grid_origin
            grid_width_px = self.cell_size * game_state.grid_width
            grid_height_px = self.cell_size * game_state.grid_height
            boundary_rect = pygame.Rect(ox, oy, grid_width_px, grid_height_px)
            pygame.draw.rect(self.screen, (255, 0, 0), boundary_rect, 2)  # Red border
        
        # Draw panels
        if info is None:
            info = {}
        self.draw_left_panel(info)
        self.draw_right_panel()
        
        pygame.display.flip()
    
    def clear_screen(self):
        """Clear the screen."""
        if self.headless or self.screen is None:
            return
        self.screen.fill(BLACK)
        pygame.display.flip()
    
    def add_leaderboard_entry(self, episode: int, score: int, death_type: str):
        """Add an entry to the leaderboard."""
        self.leaderboard.add_entry(episode, score, death_type)
        self.session_leaderboard.add_entry(episode, score, death_type)


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