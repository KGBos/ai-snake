from src.config.loader import load_config, get_grid_size, get_game_speed, get_nes_mode, get_leaderboard_file
from src.utils.logging_utils import setup_logging

setup_logging(log_to_file=False, log_to_console=True, log_level='INFO')

def test_load_config():
    config = load_config('src/config/config.yaml')
    assert 'game' in config
    assert 'ai' in config
    assert 'learning' in config
    assert 'display' in config

def test_get_grid_size():
    config = load_config('src/config/config.yaml')
    grid = get_grid_size(config)
    assert isinstance(grid, tuple)
    assert len(grid) == 2
    assert all(isinstance(x, int) for x in grid)

def test_get_game_speed():
    config = load_config('src/config/config.yaml')
    speed = get_game_speed(config)
    assert isinstance(speed, int)
    assert speed > 0

def test_get_nes_mode():
    config = load_config('src/config/config.yaml')
    nes = get_nes_mode(config)
    assert isinstance(nes, bool)

def test_get_leaderboard_file():
    config = load_config('src/config/config.yaml')
    lb_file = get_leaderboard_file(config)
    assert isinstance(lb_file, str)
    assert lb_file.endswith('.json') 