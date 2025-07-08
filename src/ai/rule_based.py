from collections import deque
from typing import Tuple, Optional, List
import logging
from src.game.models import GameState


class AIController:
    """Handles AI logic for the snake game."""
    
    def __init__(self, enable_tracing: bool = False):
        self.enable_tracing = enable_tracing
        self.move_count = 0
        self.food_eaten = 0
        self.decision_log = []
        
        if enable_tracing:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger('AI')
    
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
        snake_body = game_state.get_snake_body()
        obstacles = set(snake_body[:-1])
        
        decision_info = {
            'move_number': self.move_count,
            'head_position': head,
            'food_position': game_state.food,
            'snake_length': len(game_state.get_snake_body()),
            'strategy': None,
            'reasoning': []
        }
        
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
                direction = (next_cell[0] - head[0], next_cell[1] - head[1])
                decision_info['strategy'] = 'path_to_food'
                decision_info['reasoning'].append(f"Found safe path to food: {path[:3]}...")
                decision_info['direction'] = direction
                
                if self.enable_tracing:
                    self.logger.info(f"Move {self.move_count}: Path to food strategy - Direction: {direction}")
                
                return direction
        
        # Fallback: find direction with most open area
        best_score = None
        best_dir = None
        area_scores = {}
        
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
            area_scores[(dx, dy)] = {'area': area, 'wall_dist': wall_dist, 'score': score}
            
            if best_score is None or score > best_score:
                best_score = score
                best_dir = (dx, dy)
        
        if best_dir:
            decision_info['strategy'] = 'open_area_fallback'
            decision_info['reasoning'].append(f"Path to food blocked, using open area strategy")
            decision_info['reasoning'].append(f"Best direction: {best_dir} (score: {area_scores[best_dir]['score']:.1f})")
            decision_info['direction'] = best_dir
            
            if self.enable_tracing:
                self.logger.info(f"Move {self.move_count}: Open area fallback - Direction: {best_dir}, Score: {area_scores[best_dir]['score']:.1f}")
        
        self.decision_log.append(decision_info)
        return best_dir
    
    def make_move(self, game_state: GameState):
        """Make the AI's next move."""
        self.move_count += 1
        best_direction = self.get_best_direction(game_state)
        
        if best_direction:
            game_state.set_direction(best_direction, force=True)
        else:
            if self.enable_tracing:
                self.logger.warning(f"Move {self.move_count}: No valid move found!")
    
    def check_food_eaten(self, game_state: GameState):
        """Check if food was eaten and record it."""
        if game_state.grow:  # Food was eaten this frame
            self.food_eaten += 1
            if self.enable_tracing:
                self.logger.info(f"Food eaten! Total: {self.food_eaten}")
    
    def record_food_eaten(self):
        """Record when AI eats food."""
        self.food_eaten += 1
        if self.enable_tracing:
            self.logger.info(f"Food eaten! Total: {self.food_eaten}")
    
    def get_performance_stats(self, final_score=None) -> dict:
        """Get AI performance statistics."""
        if not self.decision_log:
            return {}
        
        strategies_used = {}
        for decision in self.decision_log:
            strategy = decision.get('strategy', 'unknown')
            strategies_used[strategy] = strategies_used.get(strategy, 0) + 1
        
        return {
            'total_moves': self.move_count,
            'food_eaten': self.food_eaten,  # Use actual food eaten count
            'strategies_used': strategies_used,
            'average_snake_length': sum(d['snake_length'] for d in self.decision_log) / len(self.decision_log) if self.decision_log else 0
        }
    
    def print_performance_report(self):
        """Print a detailed performance report."""
        if not self.enable_tracing:
            return
        
        stats = self.get_performance_stats()
        if not stats:
            return
        
        self.logger.info("=== AI Performance Report ===")
        self.logger.info(f"Total moves: {stats['total_moves']}")
        self.logger.info(f"Food eaten: {stats['food_eaten']}")
        self.logger.info(f"Strategies used: {stats['strategies_used']}")
        self.logger.info(f"Average snake length: {stats['average_snake_length']:.1f}")
        self.logger.info("===========================")

    @staticmethod
    def neighbors(pos, width, height):
        x, y = pos
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                yield nx, ny

    @staticmethod
    def bfs(start, goal, obstacles, width, height):
        """Breadth-first search returning path from start to goal (stateless)."""
        queue = deque([start])
        came_from = {start: None}
        while queue:
            current = queue.popleft()
            if current == goal:
                break
            for nxt in AIController.neighbors(current, width, height):
                if nxt in obstacles or nxt in came_from:
                    continue
                came_from[nxt] = current
                queue.append(nxt)
        if goal not in came_from:
            return None
        path = []
        node = goal
        while node != start:
            path.append(node)
            node = came_from[node]
        path.reverse()
        return path

    @staticmethod
    def path_exists_stateless(start, goal, obstacles, width, height):
        """Check whether a path exists from start to goal (stateless)."""
        queue = deque([start])
        visited = {start}
        while queue:
            current = queue.popleft()
            if current == goal:
                return True
            for nxt in AIController.neighbors(current, width, height):
                if nxt in obstacles or nxt in visited:
                    continue
                visited.add(nxt)
                queue.append(nxt)
        return False

    @staticmethod
    def open_area(start, obstacles, width, height):
        """Return number of reachable cells from start (stateless)."""
        queue = deque([start])
        visited = {start}
        count = 0
        while queue:
            x, y = queue.popleft()
            count += 1
            for nx, ny in AIController.neighbors((x, y), width, height):
                if (nx, ny) in obstacles or (nx, ny) in visited:
                    continue
                visited.add((nx, ny))
                queue.append((nx, ny))
        return count

# Stateless function for legacy compatibility (e.g., with game.py)
def ai_move(game):
    """Choose next move for the game's snake (stateless, for legacy use)."""
    head = game.snake[0]
    tail = game.snake[-1]
    snake_body = list(game.snake)
    obstacles = set(snake_body[:-1])

    path = AIController.bfs(head, game.food, obstacles, game.grid_width, game.grid_height)
    if path:
        next_cell = path[0]
        future_snake = list(game.snake)
        future_snake.insert(0, next_cell)
        if next_cell != game.food and not game.grow:
            future_snake.pop()
        new_head = next_cell
        new_tail = future_snake[-1]
        if AIController.path_exists_stateless(new_head, new_tail, set(future_snake[:-1]), game.grid_width, game.grid_height):
            game.direction = (next_cell[0] - head[0], next_cell[1] - head[1])
            return

    best_score = None
    best_dir = None
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        nx, ny = head[0] + dx, head[1] + dy
        if not (0 <= nx < game.grid_width and 0 <= ny < game.grid_height):
            continue
        if (nx, ny) in obstacles:
            continue
        area = AIController.open_area((nx, ny), set(game.snake), game.grid_width, game.grid_height)
        wall_dist = min(nx, game.grid_width - 1 - nx, ny, game.grid_height - 1 - ny)
        score = area + wall_dist * 0.5
        if best_score is None or score > best_score:
            best_score = score
            best_dir = (dx, dy)
    if best_dir:
        game.direction = best_dir

# Note: ai_legacy.py is now deprecated. Use ai.rule_based.AIController or ai.rule_based.ai_move instead. 