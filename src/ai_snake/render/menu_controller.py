import pygame
from typing import Optional, Tuple
from src.config.config import DEFAULT_GRID
from src.render.renderer import MenuRenderer


class MenuController:
    """Handles menu logic and navigation."""
    
    def __init__(self):
        self.renderer = MenuRenderer()
        self.clock = pygame.time.Clock()
    
    def run_main_menu(self) -> Optional[Tuple[dict, bool]]:
        """Run the main menu. Returns (settings, ai_mode) or None to quit."""
        selection = 0
        settings = {'speed': 10, 'grid': DEFAULT_GRID, 'nes': False}
        
        while True:
            self.renderer.render_main_menu(selection)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        selection = (selection - 1) % 4
                    elif event.key == pygame.K_DOWN:
                        selection = (selection + 1) % 4
                    elif event.key == pygame.K_RETURN:
                        if selection == 0:  # AI Game
                            return settings, True
                        elif selection == 1:  # Human Game
                            return settings, False
                        elif selection == 2:  # Settings
                            settings = self.run_settings_menu(settings)
                        elif selection == 3:  # Quit
                            return None
            
            self.clock.tick(15)
    
    def run_settings_menu(self, settings: dict) -> dict:
        """Run the settings menu. Returns updated settings."""
        selection = 0
        speeds = {0: 5, 1: 10, 2: 15}
        grids = {3: (10, 10), 4: (20, 20), 5: (30, 30)}
        
        while True:
            self.renderer.render_settings_menu(settings, selection)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return settings
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        selection = (selection - 1) % 8
                    elif event.key == pygame.K_DOWN:
                        selection = (selection + 1) % 8
                    elif event.key == pygame.K_RETURN:
                        if selection in speeds:
                            settings['speed'] = speeds[selection]
                        elif selection in grids:
                            settings['grid'] = grids[selection]
                        elif selection == 6:  # NES Style toggle
                            settings['nes'] = not settings.get('nes', False)
                        elif selection == 7:  # Back
                            return settings
            
            self.clock.tick(15)
    
    def run_pause_menu(self, game_controller) -> bool:
        """Run the pause menu. Returns True if game should continue."""
        selection = 0
        speeds = {2: 5, 3: 10, 4: 15}
        
        while True:
            self.renderer.render_pause_menu(selection)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        selection = (selection - 1) % 5
                    elif event.key == pygame.K_DOWN:
                        selection = (selection + 1) % 5
                    elif event.key == pygame.K_RETURN:
                        if selection == 0:  # Resume
                            return True
                        elif selection == 1:  # Toggle AI
                            game_controller.ai = not game_controller.ai
                        elif selection in speeds:
                            game_controller.speed = speeds[selection]
            
            self.clock.tick(15) 