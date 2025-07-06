from collections import deque
from typing import Tuple, Optional, List
from .models import GameState


class AIController:
    """Handles AI logic for the snake game."""
    
    def __init__(self):
        pass
    
    def get_neighbors(self, pos: Tuple[int, int], width: int, height: int):
        """Get valid neighboring positions."""
        x, y = pos
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                yield nx, ny
    
    def bfs_path(self, start: Tuple[int, int], goal: Tuple[int, int], 
                 obstacles: set, width: int, height: int) -> Optional[List[Tuple[int, int]]]:
        """Breadth-first search returning path from start to goal."""
        queue = deque([start])
        came_from: dict = {start: None}
        
        while queue:
            current = queue.popleft()
            if current == goal:
                break
                
            for nxt in self.get_neighbors(current, width, height):
                if nxt in obstacles or nxt in came_from:
                    continue
                came_from[nxt] = current
                queue.append(nxt)
        
        if goal not in came_from:
            return None
            
        # Reconstruct path
        path = []
        node: Optional[Tuple[int, int]] = goal
        while node is not None and node != start:
            path.append(node)
            node = came_from.get(node)
        path.reverse()
        return path
    
    def path_exists(self, start: Tuple[int, int], goal: Tuple[int, int], 
                    obstacles: set, width: int, height: int) -> bool:
        """Check whether a path exists from start to goal."""
        queue = deque([start])
        visited = {start}
        
        while queue:
            current = queue.popleft()
            if current == goal:
                return True
                
            for nxt in self.get_neighbors(current, width, height):
                if nxt in obstacles or nxt in visited:
                    continue
                visited.add(nxt)
                queue.append(nxt)
        return False
    
    def calculate_open_area(self, start: Tuple[int, int], obstacles: set, 
                           width: int, height: int) -> int:
        """Return number of reachable cells from start."""
        queue = deque([start])
        visited = {start}
        count = 0
        
        while queue:
            x, y = queue.popleft()
            count += 1
            
            for nx, ny in self.get_neighbors((x, y), width, height):
                if (nx, ny) in obstacles or (nx, ny) in visited:
                    continue
                visited.add((nx, ny))
                queue.append((nx, ny))
        return count
    
    def get_best_direction(self, game_state: GameState) -> Optional[Tuple[int, int]]:
        """Get the best direction for the AI to move."""
        head = game_state.get_snake_head()
        tail = game_state.get_snake_tail()
        obstacles = set(game_state.get_snake_body()[:-1])
        
        # Try to find path to food
        path = self.bfs_path(head, game_state.food, obstacles, 
                            game_state.grid_width, game_state.grid_height)
        
        if path:
            next_cell = path[0]
            # Simulate future snake state
            future_snake = list(game_state.get_snake_body())
            future_snake.insert(0, next_cell)
            if next_cell != game_state.food and not game_state.grow:
                future_snake.pop()
            
            new_head = next_cell
            new_tail = future_snake[-1]
            
            # Check if we can still reach tail after this move
            if self.path_exists(new_head, new_tail, set(future_snake[:-1]), 
                              game_state.grid_width, game_state.grid_height):
                return (next_cell[0] - head[0], next_cell[1] - head[1])
        
        # Fallback: find direction with most open area
        best_score = None
        best_dir = None
        
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = head[0] + dx, head[1] + dy
            
            if not game_state.is_valid_position((nx, ny)):
                continue
            if (nx, ny) in obstacles:
                continue
                
            area = self.calculate_open_area((nx, ny), set(game_state.get_snake_body()), 
                                          game_state.grid_width, game_state.grid_height)
            wall_dist = min(nx, game_state.grid_width - 1 - nx, 
                           ny, game_state.grid_height - 1 - ny)
            score = area + wall_dist * 0.5
            
            if best_score is None or score > best_score:
                best_score = score
                best_dir = (dx, dy)
        
        return best_dir
    
    def make_move(self, game_state: GameState):
        """Make the AI's next move."""
        best_direction = self.get_best_direction(game_state)
        if best_direction:
            game_state.set_direction(best_direction) 