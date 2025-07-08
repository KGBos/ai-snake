import sys
import argparse
import os
import logging
from src.utils.logging_utils import setup_logging
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pygame
from src.game_controller import GameController
from src.config.loader import load_config, get_grid_size, get_game_speed, get_nes_mode, get_auto_advance, get_ai_tracing, get_model_path

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="AI Snake Game Launcher")
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--config', type=str, default='src/config/config.yaml', help='Path to config file')
    subparsers = parser.add_subparsers(dest='command', help='Sub-commands')

    # Play subcommand
    play_parser = subparsers.add_parser('play', help='Play or watch the game')
    play_parser.add_argument('--learning', action='store_true', help='Enable learning AI mode')
    play_parser.add_argument('--model', type=str, default=None, help='Path to pre-trained model for learning AI')
    play_parser.add_argument('--ai', action='store_true', help='Enable rule-based AI mode')
    play_parser.add_argument('--speed', type=int, default=None, help='Game speed (frames per second)')
    play_parser.add_argument('--grid', type=int, nargs=2, default=None, help='Grid size, e.g. --grid 20 20')
    play_parser.add_argument('--nes', action='store_true', help='Enable NES mode (retro style)')
    play_parser.add_argument('--ai-tracing', action='store_true', help='Enable AI tracing/logging')
    play_parser.add_argument('--auto-advance', action='store_true', help='Auto-advance after game over (for training/testing)')
    play_parser.add_argument('--debug-learning', action='store_true', help='Show detailed learning progress in console')
    play_parser.add_argument('--verbose', action='store_true', help='Enable verbose logging for real-time RL observation')
    play_parser.add_argument('--headless', action='store_true', help='Run in true headless mode (no rendering)')
    play_parser.add_argument('--starvation-threshold', type=int, default=None, help='Override starvation threshold (moves without food) for this session')
    play_parser.add_argument('--web', action='store_true', help='Use web renderer (localhost Flask)')

    # Test model subcommand
    test_parser = subparsers.add_parser('test-model', help='Test a trained model')
    test_parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    test_parser.add_argument('--games', type=int, default=10, help='Number of games to run')
    test_parser.add_argument('--speed', type=int, default=None, help='Game speed')
    test_parser.add_argument('--grid', type=int, nargs=2, default=None, help='Grid size')
    test_parser.add_argument('--save', action='store_true', help='Save detailed results to file')

    # Train subcommand (future extension)
    # train_parser = subparsers.add_parser('train', help='Train the learning AI')
    # ...

    return parser.parse_args()

def play_game(args):
    # Load config
    config = load_config(args.config)
    
    # Set up logging level based on verbose flag
    if args.verbose:
        logger.info("🔍 VERBOSE LOGGING ENABLED - You'll see detailed RL progress in real-time!")
        logger.info("📊 Watch for:")
        logger.info("   🎯 Food rewards when snake eats")
        logger.info("   💀 Death penalties when snake dies")
        logger.info("   📏 Distance rewards/penalties")
        logger.info("   ⚡ Efficiency bonuses")
        logger.info("   🔄 Oscillation penalties")
        logger.info("   🛡️ Safety interventions")
        logger.info("   📈 Learning progress updates")
        logger.info("=" * 60)
    
    # Use command line args if provided, otherwise use config
    grid = tuple(args.grid) if args.grid else get_grid_size(config)
    speed = args.speed if args.speed is not None else get_game_speed(config)
    nes_mode = args.nes if args.nes else get_nes_mode(config)
    auto_advance = args.auto_advance if args.auto_advance else get_auto_advance(config)
    ai_tracing = args.ai_tracing if args.ai_tracing else get_ai_tracing(config)
    model_path = args.model if args.model else get_model_path(config)
    starvation_threshold = args.starvation_threshold if hasattr(args, 'starvation_threshold') and args.starvation_threshold is not None else None
    
    pygame.init()
    game = GameController(
        speed=speed,
        ai=args.ai,
        grid=grid,
        nes_mode=nes_mode,
        ai_tracing=ai_tracing,
        auto_advance=auto_advance,
        learning_ai=args.learning,
        model_path=model_path,
        headless=getattr(args, 'headless', False),
        web=getattr(args, 'web', False),
        starvation_threshold=starvation_threshold
    )
    
    # Enable debug learning if requested
    if args.debug_learning:
        game.debug_learning = True
    
    # Enable verbose logging for learning AI
    if args.verbose and args.learning:
        logger.info("🤖 LEARNING AI MODE WITH VERBOSE LOGGING")
        logger.info("Watch the console for detailed RL progress!")
        logger.info("Press Q to quit and see final learning report")
        logger.info("=" * 60)
    
    game.run_game_loop()
    pygame.quit()

def test_model(args):
    # Import test script from testing/
    import importlib.util
    test_path = os.path.join(os.path.dirname(__file__), '../testing/test_ai_performance.py')
    spec = importlib.util.spec_from_file_location('test_ai_performance', test_path)
    if spec is None or spec.loader is None:
        logger.error('Could not load test_ai_performance.py for model testing.')
        print('Could not load test_ai_performance.py for model testing.')
        sys.exit(1)
    test_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_module)
    
    # Load config for default values
    config = load_config(args.config)
    grid = tuple(args.grid) if args.grid else get_grid_size(config)
    speed = args.speed if args.speed is not None else get_game_speed(config)
    
    # Run test
    test_module.test_ai_performance({
        'testing': {
            'episodes': args.games,
            'speed': speed,
            'grid_size': grid
        }
    }, model_path=args.model, episodes=args.games, speed=speed)

def main():
    args = parse_args()
    # Setup centralized logging
    log_level = 'DEBUG' if getattr(args, 'debug', False) else 'INFO'
    json_mode = False  # Could add a CLI flag for this if desired
    setup_logging(log_to_file=True, log_to_console=True, log_level=log_level, json_mode=json_mode)
    logger.debug('Centralized logging enabled.')
    if args.command == 'play':
        play_game(args)
    elif args.command == 'test-model':
        test_model(args)
    else:
        logger.warning('Please specify a subcommand. Use --help for options.')
        print('Please specify a subcommand. Use --help for options.')

if __name__ == '__main__':
    main() 