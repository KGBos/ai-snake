import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from collections import deque
from src.models import GameState


class TestGameState:
    """Test cases for the GameState class."""
    
    def test_initial_state(self):
        """Test that the game state initializes correctly."""
        state = GameState(grid_width=10, grid_height=10)
        
        assert state.grid_width == 10
        assert state.grid_height == 10
        assert state.score == 0
        assert state.game_over == False
        assert state.direction == (1, 0)
        assert len(state.snake) == 1
        assert state.snake[0] == (5, 5)  # Center of 10x10 grid
    
    def test_move_snake(self):
        """Test that the snake moves correctly."""
        state = GameState(grid_width=10, grid_height=10)
        initial_head = state.get_snake_head()
        
        state.move_snake()
        
        new_head = state.get_snake_head()
        assert new_head == (initial_head[0] + 1, initial_head[1])  # Moving right
    
    def test_collision_detection(self):
        """Test collision detection."""
        state = GameState(grid_width=5, grid_height=5)
        
        # Test wall collision
        state.set_snake_for_testing([(4, 2)])  # Head at right edge
        state.direction = (1, 0)  # Moving right
        state.move_snake()  # Move into the wall
        state.check_collision(0)
        assert state.game_over == True
    
    def test_food_collection(self):
        """Test food collection and scoring."""
        state = GameState(grid_width=5, grid_height=5)
        state.set_snake_for_testing([(2, 2)])
        state.food = (3, 2)  # Food to the right
        state.direction = (1, 0)  # Moving right
        
        initial_score = state.score
        state.move_snake()
        state.check_collision(0)
        
        assert state.score == initial_score + 1
        assert state.grow == True
        assert state.food != (3, 2)  # New food spawned
    
    def test_direction_setting(self):
        """Test that direction changes work correctly."""
        state = GameState()
        state.direction = (1, 0)  # Moving right
        
        # Test valid direction change
        state.set_direction((0, -1))  # Up
        assert state.direction == (0, -1)
        
        # Test invalid 180-degree turn
        state.set_direction((0, 1))  # Down (opposite of up)
        assert state.direction == (0, -1)  # Should not change
    
    def test_reset(self):
        """Test that reset works correctly."""
        state = GameState(grid_width=10, grid_height=10)
        state.score = 100
        state.game_over = True
        state.set_snake_for_testing([(1, 1), (2, 1), (3, 1)])
        
        state.reset()
        
        assert state.score == 0
        assert state.game_over == False
        assert len(state.snake) == 1
        assert state.snake[0] == (5, 5)  # Back to center
        assert state.direction == (1, 0)  # Back to initial direction


def run_tests():
    """Run all tests."""
    test_instance = TestGameState()
    
    print("Running GameState tests...")
    
    try:
        test_instance.test_initial_state()
        print("✓ test_initial_state passed")
    except Exception as e:
        print(f"✗ test_initial_state failed: {e}")
    
    try:
        test_instance.test_move_snake()
        print("✓ test_move_snake passed")
    except Exception as e:
        print(f"✗ test_move_snake failed: {e}")
    
    try:
        test_instance.test_collision_detection()
        print("✓ test_collision_detection passed")
    except Exception as e:
        print(f"✗ test_collision_detection failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_instance.test_food_collection()
        print("✓ test_food_collection passed")
    except Exception as e:
        print(f"✗ test_food_collection failed: {e}")
    
    try:
        test_instance.test_direction_setting()
        print("✓ test_direction_setting passed")
    except Exception as e:
        print(f"✗ test_direction_setting failed: {e}")
    
    try:
        test_instance.test_reset()
        print("✓ test_reset passed")
    except Exception as e:
        print(f"✗ test_reset failed: {e}")
    
    print("All tests completed!")


if __name__ == '__main__':
    run_tests() 