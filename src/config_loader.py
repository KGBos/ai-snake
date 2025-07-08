import yaml
import os
from typing import Dict, Any, Tuple

def load_config(config_path: str = 'src/config.yaml') -> Dict[str, Any]:
    """Load YAML config file and return as a dictionary."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_grid_size(config: Dict[str, Any]) -> Tuple[int, int]:
    """Get grid size from config."""
    return (config['game']['grid_width'], config['game']['grid_height'])

def get_game_speed(config: Dict[str, Any]) -> int:
    """Get game speed from config."""
    return config['game']['speed']

def get_nes_mode(config: Dict[str, Any]) -> bool:
    """Get NES mode setting from config."""
    return config['game']['nes_mode']

def get_auto_advance(config: Dict[str, Any]) -> bool:
    """Get auto advance setting from config."""
    return config['game']['auto_advance']

def get_ai_tracing(config: Dict[str, Any]) -> bool:
    """Get AI tracing setting from config."""
    return config['ai']['enable_tracing']

def get_model_path(config: Dict[str, Any]) -> str:
    """Get model path from config."""
    return config['ai']['model_path']

def get_screen_size(config: Dict[str, Any]) -> Tuple[int, int]:
    """Get screen size from config."""
    return (config['display']['screen_width'], config['display']['screen_height'])

def get_stats_area_height(config: Dict[str, Any]) -> int:
    """Get stats area height from config."""
    return config['display']['stats_area_height']

def get_grid_padding(config: Dict[str, Any]) -> int:
    """Get grid padding from config (pixels)."""
    return config['display'].get('grid_padding', 0)

def get_panel_padding(config: Dict[str, Any]) -> int:
    """Get panel padding from config (pixels)."""
    return config['display'].get('panel_padding', 15)

def get_leaderboard_file(config: Dict[str, Any]) -> str:
    """Get leaderboard file path from config."""
    return config['display'].get('leaderboard_file', 'leaderboard.json') 