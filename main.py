import pygame
from snake.game_controller import GameController
from snake.menu_controller import MenuController
from snake.config import DEFAULT_GRID


def main():
    """Main entry point for the AI Snake game."""
    pygame.init()
    
    menu_controller = MenuController()
    
    while True:
        # Run main menu
        result = menu_controller.run_main_menu()
        if result is None:
            break
        
        settings, ai_mode = result
        
        # Create and configure game controller
        game_controller = GameController(
            speed=settings['speed'],
            ai=ai_mode,
            grid=settings.get('grid', DEFAULT_GRID),
            nes_mode=settings.get('nes', False)
        )
        
        # Run game loop
        if not game_controller.run_game_loop():
            break
    
    pygame.quit()


if __name__ == '__main__':
    main()
