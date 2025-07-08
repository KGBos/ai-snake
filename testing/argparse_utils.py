import argparse
import os
import logging
from typing import Any, Dict, Optional
from src.config_loader import load_config

def load_testing_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """Load the 'testing' section from config.yaml as defaults."""
    config = load_config(config_path)
    return config.get('testing', {})

def get_base_parser(defaults: Optional[Dict[str, Any]] = None) -> argparse.ArgumentParser:
    """Return an ArgumentParser with common testing arguments and defaults."""
    if defaults is None:
        defaults = {}
    parser = argparse.ArgumentParser(description="Unified Snake AI Performance Tester")
    parser.add_argument('--model', type=str, default=defaults.get('model'), help='Path to trained model')
    parser.add_argument('--games', type=int, default=defaults.get('games', 50), help='Number of games to run')
    parser.add_argument('--speed', type=int, default=defaults.get('speed', 60), help='Game speed')
    parser.add_argument('--grid', type=int, nargs=2, default=defaults.get('grid_size', [20, 20]), help='Grid size')
    parser.add_argument('--compare', action='store_true', default=defaults.get('compare', False), help='Compare trained vs untrained')
    parser.add_argument('--visual', action='store_true', default=defaults.get('visual', False), help='Show games visually')
    parser.add_argument('--save', action='store_true', help='Save results to file')
    parser.add_argument('--auto-advance', type=lambda x: (str(x).lower() == 'true'), default=defaults.get('auto_advance', True), help='Automatically advance to next game (true/false)')
    parser.add_argument('--advanced-report', action='store_true', help='Enable advanced statistics and reporting')
    return parser

def parse_args_with_config(config_path: str = 'config.yaml') -> argparse.Namespace:
    """Load config defaults, parse CLI args, and merge them (CLI takes precedence)."""
    defaults = load_testing_config(config_path)
    parser = get_base_parser(defaults)
    args = parser.parse_args()
    return args

def setup_logging_from_config(config_path: str = 'config.yaml'):
    """Set up logging based on the logging section of config.yaml."""
    if not os.path.exists(config_path):
        return
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    log_cfg = config.get('logging', {})
    level = getattr(logging, log_cfg.get('level', 'INFO').upper(), logging.INFO)
    log_format = log_cfg.get('format', '[%(levelname)s] %(message)s')
    if log_cfg.get('to_file', False):
        logging.basicConfig(level=level, format=log_format, filename=log_cfg.get('filename', 'performance.log'), filemode='w')
    else:
        logging.basicConfig(level=level, format=log_format) 