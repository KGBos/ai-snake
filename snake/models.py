from collections import deque
from dataclasses import dataclass, field
from typing import Tuple, Optional, Deque
import random


@dataclass
class GameState:
    """Represents the current state of the snake game."""
    grid_width: int = 20
    grid_height: int = 20
    snake: Deque[Tuple[int, int]] = field(default_factory=deque)
    food: Tuple[int, int] = (0, 0)
    direction: Tuple[int, int] = (1, 0)
    score: int = 0
    high_score: int = 0
    grow: bool = False
    game_over: bool = False
    last_food_time: Optional[int] = None
    
    def __post_init__(self):
        """Initialize the game state after creation."""
        self.reset()
    
    def reset(self):
        """Reset the game to initial state."""
        self.direction = (1, 0)
        self.snake = deque([(self.grid_width // 2, self.grid_height // 2)])
        self.spawn_food()
        self.grow = False
        self.game_over = False
        self.score = 0
        self.last_food_time = None
    
    def spawn_food(self):
        """Spawn food at a random location not occupied by the snake."""
        if len(self.snake) >= self.grid_width * self.grid_height:
            self.game_over = True
            return
        
        while True:
            self.food = (
                random.randint(0, self.grid_width - 1),
                random.randint(0, self.grid_height - 1)
            )
            if self.food not in self.snake:
                break
    
    def move_snake(self):
        """Move the snake in the current direction."""
        hx, hy = self.snake[0]
        dx, dy = self.direction
        nx, ny = hx + dx, hy + dy
        self.snake.appendleft((nx, ny))
    
    def check_collision(self, current_time: int):
        """Check for collisions and handle food collection."""
        hx, hy = self.snake[0]
        
        # Check wall collision
        if not (0 <= hx < self.grid_width and 0 <= hy < self.grid_height):
            self.game_over = True
            return
        
        # Check self collision
        if (hx, hy) in list(self.snake)[1:]:
            self.game_over = True
            return
        
        # Check food collision
        if (hx, hy) == self.food:
            self.grow = True
            if self.last_food_time and current_time - self.last_food_time <= 3000:
                self.score += 2
            else:
                self.score += 1
            self.last_food_time = current_time
            self.spawn_food()
    
    def handle_growth(self):
        """Handle snake growth after eating food."""
        if not self.grow:
            self.snake.pop()
        else:
            self.grow = False
    
    def set_direction(self, new_direction: Tuple[int, int]):
        """Set the snake's direction if it's valid."""
        # Prevent 180-degree turns
        if (new_direction[0] != -self.direction[0] or 
            new_direction[1] != -self.direction[1]):
            self.direction = new_direction
    
    def get_snake_head(self) -> Tuple[int, int]:
        """Get the snake's head position."""
        return self.snake[0]
    
    def get_snake_tail(self) -> Tuple[int, int]:
        """Get the snake's tail position."""
        return self.snake[-1]
    
    def get_snake_body(self) -> list:
        """Get the snake's body as a list."""
        return list(self.snake)
    
    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is within the grid bounds."""
        x, y = pos
        return 0 <= x < self.grid_width and 0 <= y < self.grid_height
    
    def set_snake_for_testing(self, snake_body: list):
        """Set the snake body for testing purposes."""
        self.snake = deque(snake_body) 