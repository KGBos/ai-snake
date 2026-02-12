import argparse
import importlib.util
import logging
import os
import subprocess
import sys

from ai_snake.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Show defaults and preserve multiline example blocks."""


def setup_ai_logging(log_to_file):
    if log_to_file:
        os.makedirs("logs", exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            filename="logs/ai_moves.log",
            filemode="w",
            format="%(asctime)s [%(levelname)s] %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="AI Snake Game Launcher",
        formatter_class=_HelpFormatter,
        epilog=(
            "Examples:\n"
            "  snake play --learning --headless\n"
            "  snake play --ai --speed 20\n"
            "  snake test-model --model models/snake_dqn_model.pth --games 25\n"
            "  snake eval-model --model snake_dqn_model_auto.pth --episodes 200\n"
        ),
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    play_parser = subparsers.add_parser(
        "play",
        help="Run the game (manual, rule-based AI, or learning AI)",
        formatter_class=_HelpFormatter,
    )
    play_parser.add_argument("--learning", action="store_true", help="Enable learning AI mode")
    play_parser.add_argument("--model", type=str, default=None, help="Path to pre-trained model for learning AI")
    play_parser.add_argument("--ai", action="store_true", help="Enable rule-based AI mode")
    play_parser.add_argument("--speed", type=int, default=None, help="Game speed (frames per second)")
    play_parser.add_argument("--grid", type=int, nargs=2, default=None, help="Grid size, e.g. --grid 20 20")
    play_parser.add_argument("--nes", action="store_true", help="Enable NES mode (retro style)")
    play_parser.add_argument("--ai-tracing", action="store_true", help="Enable AI tracing/logging")
    play_parser.add_argument("--auto-advance", action="store_true", help="Auto-advance after game over (for training/testing)")
    play_parser.add_argument("--debug-learning", action="store_true", help="Show detailed learning progress in console")
    play_parser.add_argument("--verbose", action="store_true", help="Enable verbose logging for real-time RL observation")
    play_parser.add_argument("--headless", action="store_true", help="Run in true headless mode (no rendering)")
    play_parser.add_argument("--starvation-threshold", type=int, default=None, help="Override starvation threshold (moves without food) for this session")
    play_parser.add_argument("--web", action="store_true", help="Use web renderer (localhost Flask)")
    play_parser.add_argument("--log", action="store_true", help="Save AI logs with timestamps to logs/ai_moves.log")

    test_parser = subparsers.add_parser(
        "test-model",
        help="Run model performance test harness",
        formatter_class=_HelpFormatter,
    )
    test_parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    test_parser.add_argument("--games", type=int, default=10, help="Number of games to run")
    test_parser.add_argument("--speed", type=int, default=None, help="Game speed")
    test_parser.add_argument("--grid", type=int, nargs=2, default=None, help="Grid size")
    test_parser.add_argument("--save", action="store_true", help="Save detailed results to file")

    eval_parser = subparsers.add_parser(
        "eval-model",
        help="Evaluate one or more model checkpoints and export CSV/JSON metrics",
        formatter_class=_HelpFormatter,
        epilog=(
            "Examples:\n"
            "  snake eval-model --model snake_dqn_model_auto.pth --episodes 100\n"
            "  snake eval-model --model m1.pth m2.pth --episodes 200 --seed 123\n"
            "  snake eval-model --model m1.pth --grid 20 20 --max-steps 1500 --output-prefix outputs/eval_run\n"
        ),
    )
    eval_parser.add_argument("--model", nargs="+", required=True, help="One or more model checkpoint paths")
    eval_parser.add_argument("--episodes", type=int, default=100, help="Episodes per model")
    eval_parser.add_argument("--seed", type=int, default=42, help="Base seed for reproducible evaluation")
    eval_parser.add_argument("--grid", type=int, nargs=2, default=None, help="Grid size override")
    eval_parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    eval_parser.add_argument("--starvation-threshold", type=int, default=50, help="Moves without food before starvation death")
    eval_parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], default=None, help="Torch device override")
    eval_parser.add_argument("--output-prefix", type=str, default=None, help="Output prefix for CSV/JSON results")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        parser.exit(2)
    return args


def play_game(args):
    import pygame
    from ai_snake.config.loader import (
        get_ai_tracing,
        get_auto_advance,
        get_game_speed,
        get_grid_size,
        get_model_path,
        get_nes_mode,
        load_config,
    )
    from ai_snake.game.game_controller import GameController

    config = load_config(args.config)
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

    grid = tuple(args.grid) if args.grid else get_grid_size(config)
    speed = args.speed if args.speed is not None else get_game_speed(config)
    nes_mode = args.nes if args.nes else get_nes_mode(config)
    auto_advance = args.auto_advance if args.auto_advance else get_auto_advance(config)
    ai_tracing = args.ai_tracing if args.ai_tracing else get_ai_tracing(config)
    if args.model:
        model_path = args.model
        logger.info(f"Loading model from command line argument: {model_path}")
    else:
        model_path = get_model_path(config)
        logger.info(f"Loading model from config file: {model_path}")
    starvation_threshold = args.starvation_threshold if args.starvation_threshold is not None else None

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
        headless=getattr(args, "headless", False),
        web=getattr(args, "web", False),
        starvation_threshold=starvation_threshold,
        log_to_file=getattr(args, "log", False),
    )
    if args.debug_learning:
        game.debug_learning = True
    if args.verbose and args.learning:
        logger.info("🤖 LEARNING AI MODE WITH VERBOSE LOGGING")
        logger.info("Watch the console for detailed RL progress!")
        logger.info("Press Q to quit and see final learning report")
        logger.info("=" * 60)

    game.run_game_loop()
    pygame.quit()


def test_model(args):
    from ai_snake.config.loader import get_game_speed, get_grid_size, load_config

    test_path = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "test_ai_performance.py")
    test_path = os.path.abspath(test_path)
    spec = importlib.util.spec_from_file_location("test_ai_performance", test_path)
    if spec is None or spec.loader is None:
        logger.error("Could not load test_ai_performance.py for model testing.")
        print("Could not load test_ai_performance.py for model testing.")
        sys.exit(1)
    test_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_module)

    config = load_config(args.config)
    grid = tuple(args.grid) if args.grid else get_grid_size(config)
    speed = args.speed if args.speed is not None else get_game_speed(config)
    test_module.test_ai_performance(
        {
            "testing": {
                "episodes": args.games,
                "speed": speed,
                "grid_size": grid,
            }
        },
        model_path=args.model,
        episodes=args.games,
        speed=speed,
    )


def eval_model(args):
    eval_script = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "eval_model.py")
    eval_script = os.path.abspath(eval_script)
    cmd = [
        sys.executable,
        eval_script,
        "--model",
        *args.model,
        "--episodes",
        str(args.episodes),
        "--seed",
        str(args.seed),
        "--max-steps",
        str(args.max_steps),
        "--starvation-threshold",
        str(args.starvation_threshold),
        "--config",
        args.config,
    ]
    if args.grid:
        cmd.extend(["--grid", str(args.grid[0]), str(args.grid[1])])
    if args.device:
        cmd.extend(["--device", args.device])
    if args.output_prefix:
        cmd.extend(["--output-prefix", args.output_prefix])

    logger.info("Running model evaluation benchmark...")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Model evaluation failed with exit code {e.returncode}")
        sys.exit(e.returncode)


def main():
    args = parse_args()
    log_level = "DEBUG" if getattr(args, "debug", False) else "INFO"
    json_mode = False
    setup_ai_logging(getattr(args, "log", False))
    setup_logging(log_to_file=True, log_to_console=True, log_level=log_level, json_mode=json_mode)
    logger.debug("Centralized logging enabled.")
    if args.command == "play":
        play_game(args)
    elif args.command == "test-model":
        test_model(args)
    elif args.command == "eval-model":
        eval_model(args)
    else:
        logger.warning("Please specify a subcommand. Use --help for options.")
        print("Please specify a subcommand. Use --help for options.")


if __name__ == "__main__":
    main()
