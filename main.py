import pygame
import argparse
import sys
from snake.game_controller import GameController
from snake.menu_controller import MenuController
from snake.config import DEFAULT_GRID


def main():
    """Main entry point for the AI Snake game."""
    pygame.init()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AI Snake Game with Learning')
    parser.add_argument('--learning', action='store_true', help='Enable learning AI mode')
    parser.add_argument('--model', type=str, help='Path to pre-trained model file')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--speed', type=int, default=10, help='Game speed')
    parser.add_argument('--auto-advance', action='store_true', help='Auto-advance after game over')
    parser.add_argument('--grid', type=int, nargs=2, default=DEFAULT_GRID, help='Grid size (width height)')
    
    args = parser.parse_args()
    
    if args.learning:
        # Learning AI mode
        print("Starting Learning AI Mode...")
        print(f"Episodes: {args.episodes}")
        print(f"Model path: {args.model}")
        print("Controls:")
        print("  L - Toggle learning AI on/off")
        print("  S - Save model")
        print("  ESC - Quit")
        
        game_controller = GameController(
            speed=args.speed,
            ai=False,
            learning_ai=True,
            grid=tuple(args.grid),
            auto_advance=args.auto_advance,
            model_path=args.model
        )
        
        # Run training episodes
        episode = 0
        while episode < args.episodes:
            if not game_controller.run_game_loop():
                break
            episode += 1
            
            # Save model periodically
            if episode % 100 == 0:
                game_controller.save_model()
                print(f"Saved model at episode {episode}")
        
        # Final save
        game_controller.save_model()
        print("Training complete!")
        
    else:
        # Normal menu mode
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
                nes_mode=settings.get('nes', False),
                ai_tracing=settings.get('ai_tracing', False)  # Enable AI tracing for debugging
            )
            
            # Run game loop
            if not game_controller.run_game_loop():
                break
    
    pygame.quit()


if __name__ == '__main__':
    main()
