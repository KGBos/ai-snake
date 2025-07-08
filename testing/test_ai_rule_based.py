from ai_snake.ai.rule_based import AIController
from ai_snake.game.models import GameState
from ai_snake.utils.logging_utils import setup_logging

setup_logging(log_to_file=False, log_to_console=True, log_level='INFO')

def test_path_to_food():
    state = GameState(grid_width=5, grid_height=5)
    state.set_snake_for_testing([(2, 2)])
    state.food = (4, 2)
    ai = AIController()
    direction = ai.get_best_direction(state)
    assert direction in [(1, 0), (0, 1), (0, -1), (-1, 0)]

def test_no_path_blocked():
    state = GameState(grid_width=3, grid_height=3)
    state.set_snake_for_testing([(1, 1), (1, 0), (0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1)])
    state.food = (2, 0)
    ai = AIController()
    direction = ai.get_best_direction(state)
    assert direction == (1, 0)

def test_open_area_fallback():
    state = GameState(grid_width=4, grid_height=4)
    state.set_snake_for_testing([(0, 0)])
    state.food = (3, 3)
    ai = AIController()
    direction = ai.get_best_direction(state)
    assert direction in [(1, 0), (0, 1)]

def test_no_valid_move():
    state = GameState(grid_width=2, grid_height=2)
    state.set_snake_for_testing([(0, 0), (0, 1), (1, 1)])
    state.food = (1, 0)
    ai = AIController()
    direction = ai.get_best_direction(state)
    assert direction == (1, 0) 