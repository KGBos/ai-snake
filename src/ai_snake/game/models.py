# FIXME: Review this file for potential issues or improvements
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
    death_type: Optional[str] = None  # Track cause of death
    
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
        self.death_type = None  # Reset death type
    
    def spawn_food(self):
        """Spawn food at a random location not occupied by the snake."""
        total_cells = self.grid_width * self.grid_height
        if len(self.snake) >= total_cells:
            self.game_over = True
            self.death_type = 'grid_full'  # Snake filled entire grid
            return

        # When the snake is small, random retries are fast (few collisions).
        # When the snake is large, compute the set of free cells instead.
        if len(self.snake) < total_cells * 0.5:
            snake_set = set(self.snake)
            while True:
                pos = (
                    random.randint(0, self.grid_width - 1),
                    random.randint(0, self.grid_height - 1),
                )
                if pos not in snake_set:
                    self.food = pos
                    return
        else:
            snake_set = set(self.snake)
            free_cells = [
                (x, y)
                for x in range(self.grid_width)
                for y in range(self.grid_height)
                if (x, y) not in snake_set
            ]
            self.food = random.choice(free_cells)
    
    def move_snake(self):
        """Move the snake in the current direction."""
        hx, hy = self.snake[0]
        dx, dy = self.direction
        nx, ny = hx + dx, hy + dy
        
        # Check for wall collision immediately
        if not (0 <= nx < self.grid_width and 0 <= ny < self.grid_height):
            self.game_over = True
            self.death_type = 'wall'
            return
        
        # Check for self collision immediately
        snake_body = list(self.snake)
        if (nx, ny) in snake_body[1:]:
            self.game_over = True
            self.death_type = 'self'
            return
        
        # Add new head
        self.snake.appendleft((nx, ny))
        
        # Remove tail if not growing
        if not self.grow:
            self.snake.pop()
        else:
            self.grow = False  # Reset grow flag
    
    def check_collision(self, current_time: int):
        """Check for collisions and handle food collection."""
        hx, hy = self.snake[0]
        
        # Collision already checked in move_snake, just check food
        # Check food collision
        if (hx, hy) == self.food:
            self.grow = True
            if self.last_food_time and current_time - self.last_food_time <= 3000:
                self.score += 2
            else:
                self.score += 1
            self.last_food_time = current_time
            self.spawn_food()
    
    def set_direction(self, new_direction: Tuple[int, int], force: bool = False):
        """Set the snake's direction if it's valid."""
        # Normalize directions to prevent subtle bugs
        if not isinstance(new_direction, tuple) or len(new_direction) != 2:
            return
        ndx, ndy = new_direction
        cdx, cdy = self.direction
        # Only allow 180-degree turn if forced
        if force or (ndx, ndy) != (-cdx, -cdy):
            self.direction = (ndx, ndy)
    
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

    def set_starvation_death(self):
        """Set the game over state and death type for starvation."""
        self.game_over = True
        self.death_type = 'starvation'

    def set_death_type(self, death_type: str):
        """Set the game over state and death type for a custom reason."""
        self.game_over = True
        self.death_type = death_type

    def set_other_death(self):
        """Set the game over state and death type as 'other' for unknown causes."""
        self.game_over = True
        self.death_type = 'other' 