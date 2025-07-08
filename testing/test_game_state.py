from src.game.models import GameState
from src.utils.logging_utils import setup_logging

setup_logging(log_to_file=False, log_to_console=True, log_level='INFO')

def test_initial_state():
    state = GameState(grid_width=10, grid_height=10)
    assert state.grid_width == 10
    assert state.grid_height == 10
    assert state.score == 0
    assert state.game_over is False
    assert state.direction == (1, 0)
    assert len(state.snake) == 1
    assert state.snake[0] == (5, 5)

def test_move_snake():
    state = GameState(grid_width=10, grid_height=10)
    initial_head = state.get_snake_head()
    state.move_snake()
    new_head = state.get_snake_head()
    assert new_head == (initial_head[0] + 1, initial_head[1])

def test_collision_detection():
    state = GameState(grid_width=5, grid_height=5)
    state.set_snake_for_testing([(4, 2)])
    state.direction = (1, 0)
    state.move_snake()
    state.check_collision(0)
    assert state.game_over is True

def test_food_collection():
    state = GameState(grid_width=5, grid_height=5)
    state.set_snake_for_testing([(2, 2)])
    state.food = (3, 2)
    state.direction = (1, 0)
    initial_score = state.score
    state.move_snake()
    state.check_collision(0)
    assert state.score == initial_score + 1
    assert state.grow is True
    assert state.food != (3, 2)

def test_direction_setting():
    state = GameState()
    state.direction = (1, 0)
    state.set_direction((0, -1))
    assert state.direction == (0, -1)
    state.set_direction((0, 1))
    assert state.direction == (0, -1)

def test_reset():
    state = GameState(grid_width=10, grid_height=10)
    state.score = 100
    state.game_over = True
    state.set_snake_for_testing([(1, 1), (2, 1), (3, 1)])
    state.reset()
    assert state.score == 0
    assert state.game_over is False
    assert len(state.snake) == 1
    assert state.snake[0] == (5, 5)
    assert state.direction == (1, 0) 