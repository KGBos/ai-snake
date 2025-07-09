import os
import sys
from collections import deque
from typing import Tuple, Optional, List, Set, Dict, Any
import logging
from ai_snake.game.models import GameState
import random
from copy import deepcopy

# Utility to check for --log flag in sys.argv
LOG_TO_FILE = '--log' in sys.argv

class AIController:
    """Handles AI logic for the snake game."""
    WALL_DIST_WEIGHT = 0.5  # Heuristic weight for wall distance
    STARVATION_LIMIT = 30   # Moves without food before forced risk
    LOOP_HISTORY_SIZE = 20
    LOOP_REPEAT_THRESHOLD = 4

    def __init__(self, enable_tracing: bool = False, log_to_file: Optional[bool] = None):
        self.enable_tracing = enable_tracing
        self.move_count = 0
        self.food_eaten = 0
        self.decision_log: List[Dict[str, Any]] = []
        self.logger = logging.getLogger('AI')
        self.moves_since_food = 0
        self.loop_history = []
        self.turns_this_game = 0

    def get_neighbors(self, pos: Tuple[int, int], width: int, height: int) -> List[Tuple[int, int]]:
        """Get valid neighboring positions."""
        x, y = pos
        neighbors = []
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                neighbors.append((nx, ny))
        return neighbors

    def bfs_path(self, start: Tuple[int, int], goal: Tuple[int, int], 
                 obstacles: Set[Tuple[int, int]], width: int, height: int) -> Optional[List[Tuple[int, int]]]:
        """Breadth-first search returning path from start to goal."""
        queue = deque([start])
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
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
                    obstacles: Set[Tuple[int, int]], width: int, height: int) -> bool:
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

    def calculate_open_area(self, start: Tuple[int, int], obstacles: Set[Tuple[int, int]], 
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

    def is_path_safe(self, game_state, path, lookahead=5):
        """Simulate following the path for 'lookahead' steps and check if tail remains reachable."""
        simulated_state = deepcopy(game_state)
        for step in range(min(lookahead, len(path))):
            next_cell = path[step]
            simulated_state.snake.appendleft(next_cell)
            if next_cell != simulated_state.food and not simulated_state.grow:
                simulated_state.snake.pop()
            simulated_state.grow = False
            if not self.path_exists(simulated_state.get_snake_head(), simulated_state.get_snake_tail(), set(simulated_state.get_snake_body()[:-1]), simulated_state.grid_width, simulated_state.grid_height):
                return False
        return True

    def is_creating_box(self, game_state, next_cell):
        """Detect if moving to next_cell would create a tight box/loop (3 sides blocked)."""
        x, y = next_cell
        blocked = 0
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = x+dx, y+dy
            if not (0 <= nx < game_state.grid_width and 0 <= ny < game_state.grid_height):
                blocked += 1
            elif (nx, ny) in game_state.get_snake_body():
                blocked += 1
        return blocked >= 3

    def get_path_diversity(self, paths):
        """Return a list of paths with maximal diversity (spread out moves)."""
        if not paths:
            return []
        # For now, shuffle to avoid always picking the same path
        random.shuffle(paths)
        return paths

    def get_best_direction(self, game_state: GameState) -> Optional[Tuple[int, int]]:
        head = game_state.get_snake_head()
        tail = game_state.get_snake_tail()
        snake_body = game_state.get_snake_body()
        obstacles = set(snake_body[:-1])
        food = game_state.food
        grid_w, grid_h = game_state.grid_width, game_state.grid_height
        decision_info: Dict[str, Any] = {
            'move_number': self.move_count,
            'head_position': head,
            'food_position': food,
            'snake_length': len(snake_body),
            'strategy': None,
            'reasoning': [],
            'direction': None
        }
        # Endgame: if snake is very long, prioritize survival
        endgame = len(snake_body) > (grid_w * grid_h) * 0.7
        # 1. Try to find all shortest paths to food
        all_paths = []
        path = self.bfs_path(head, food, obstacles, grid_w, grid_h)
        if path:
            all_paths.append(path)
        # 2. Path diversity: shuffle or prioritize center
        all_paths = self.get_path_diversity(all_paths)
        # 3. Lookahead: only consider safe paths
        starvation = self.moves_since_food >= self.STARVATION_LIMIT
        # Loop detection
        self.loop_history.append(game_state.get_snake_head())
        if len(self.loop_history) > self.LOOP_HISTORY_SIZE:
            self.loop_history.pop(0)
        head_counts = {pos: self.loop_history.count(pos) for pos in set(self.loop_history)}
        in_loop = any(count >= self.LOOP_REPEAT_THRESHOLD for count in head_counts.values())
        safe_paths = [p for p in all_paths if self.is_path_safe(game_state, p, lookahead=7 if endgame else 4)]
        risky_path = all_paths[0] if all_paths and not safe_paths else None
        if safe_paths:
            chosen_path = safe_paths[0]
            next_cell = chosen_path[0]
            # Avoid creating a box/loop
            if not self.is_creating_box(game_state, next_cell):
                direction = (next_cell[0] - head[0], next_cell[1] - head[1])
                decision_info['strategy'] = 'path_to_food_safe'
                decision_info['reasoning'].append(f"Safe path to food: {chosen_path[:3]}...")
                decision_info['direction'] = direction
                if self.enable_tracing:
                    self.logger.info(f"Move {self.move_count}: Safe path to food strategy - Direction: {direction}")
                self.decision_log.append(decision_info)
                return direction
        # Starvation or loop: take a risk if needed
        if (starvation or in_loop) and risky_path:
            next_cell = risky_path[0]
            direction = (next_cell[0] - head[0], next_cell[1] - head[1])
            reason = f"Starved {self.moves_since_food} moves" if starvation else "Loop detected"
            decision_info['strategy'] = 'forced_risky_food'
            decision_info['reasoning'].append(f"{reason}, taking risky path to food: {risky_path[:3]}...")
            decision_info['direction'] = direction
            if self.enable_tracing:
                self.logger.warning(f"Move {self.move_count}: {reason} - Risky food attempt: {direction}")
            self.decision_log.append(decision_info)
            return direction
        # 4. Tail-chasing: if food path is unsafe, follow tail
        tail_path = self.bfs_path(head, tail, obstacles, grid_w, grid_h)
        if tail_path and len(tail_path) > 0:
            next_cell = tail_path[0]
            if not self.is_creating_box(game_state, next_cell):
                direction = (next_cell[0] - head[0], next_cell[1] - head[1])
                decision_info['strategy'] = 'tail_chase'
                decision_info['reasoning'].append(f"Chasing tail for safety: {tail_path[:3]}...")
                decision_info['direction'] = direction
                if self.enable_tracing:
                    self.logger.info(f"Move {self.move_count}: Tail-chasing strategy - Direction: {direction}")
                self.decision_log.append(decision_info)
                return direction
        # 5. Aggressive open area fallback
        best_score = None
        best_dir = None
        area_scores = {}
        valid_dirs = []
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = head[0] + dx, head[1] + dy
            if not game_state.is_valid_position((nx, ny)):
                continue
            if (nx, ny) in obstacles:
                continue
            valid_dirs.append((dx, dy))
            if self.is_creating_box(game_state, (nx, ny)):
                continue
            area = self.calculate_open_area((nx, ny), set(snake_body), grid_w, grid_h)
            wall_dist = min(nx, grid_w - 1 - nx, ny, grid_h - 1 - ny)
            # Endgame: weight open area more
            score = area * (2.0 if endgame else 1.0) + wall_dist * self.WALL_DIST_WEIGHT
            area_scores[(dx, dy)] = {'area': area, 'wall_dist': wall_dist, 'score': score}
            if best_score is None or score > best_score:
                best_score = score
                best_dir = (dx, dy)
        if best_dir:
            decision_info['strategy'] = 'open_area_fallback'
            decision_info['reasoning'].append("Path to food blocked, using open area strategy")
            decision_info['reasoning'].append(f"Best direction: {best_dir} (score: {area_scores[best_dir]['score']:.1f})")
            decision_info['direction'] = best_dir
            if self.enable_tracing:
                self.logger.info(f"Move {self.move_count}: Open area fallback - Direction: {best_dir}, Score: {area_scores[best_dir]['score']:.1f}")
            self.decision_log.append(decision_info)
            return best_dir
        # If all else fails, pick any valid move (even if unsafe)
        if valid_dirs:
            fallback_dir = valid_dirs[0]
            decision_info['strategy'] = 'last_resort_any_valid_move'
            decision_info['reasoning'].append("No safe move found, taking any available move.")
            decision_info['direction'] = fallback_dir
            if self.enable_tracing:
                self.logger.warning(f"Move {self.move_count}: No safe move, taking any available move: {fallback_dir}")
            self.decision_log.append(decision_info)
            return fallback_dir
        # No valid moves at all
        decision_info['strategy'] = 'no_valid_move'
        decision_info['reasoning'].append("No valid move found!")
        if self.enable_tracing:
            self.logger.warning(f"Move {self.move_count}: No valid move found!")
        self.decision_log.append(decision_info)
        return None

    def make_move(self, game_state: Any) -> None:
        """Make the AI's next move."""
        self.move_count += 1
        self.turns_this_game += 1
        best_direction = self.get_best_direction(game_state)
        if best_direction:
            game_state.set_direction(best_direction, force=True)
        # Log death type if game is over
        if self.enable_tracing and getattr(game_state, 'game_over', False):
            death_type = getattr(game_state, 'death_type', None)
            self.logger.info(f"Game over at move {self.move_count}. Death type: {death_type}")
        # Starvation/loop counter update
        if getattr(game_state, 'grow', False):
            self.moves_since_food = 0
            self.loop_history = []
        else:
            self.moves_since_food += 1
        # Log if 50 turns reached
        if self.turns_this_game == 50 and self.enable_tracing:
            self.logger.info("Reached 50 turns in this game.")

    def check_food_eaten(self, game_state: GameState) -> None:
        """Check if food was eaten and record it."""
        if game_state.grow:  # Food was eaten this frame
            self.food_eaten += 1
            if self.enable_tracing:
                self.logger.info(f"Food eaten! Total: {self.food_eaten}")

    def record_food_eaten(self) -> None:
        """Record when AI eats food."""
        self.food_eaten += 1
        if self.enable_tracing:
            self.logger.info(f"Food eaten! Total: {self.food_eaten}")

    def log_death(self, death_type, move_count):
        if self.enable_tracing:
            self.logger.info(f"Game over at move {move_count}. Death type: {death_type}")

    def reset_stats(self):
        self.move_count = 0
        self.food_eaten = 0
        self.decision_log = []
        self.moves_since_food = 0
        self.loop_history = []
        self.turns_this_game = 0

    def get_performance_stats(self, final_score: Optional[int] = None) -> dict:
        """Get AI performance statistics."""
        if not self.decision_log:
            return {}
        strategies_used = {}
        for decision in self.decision_log:
            strategy = decision.get('strategy', 'unknown')
            strategies_used[strategy] = strategies_used.get(strategy, 0) + 1
        return {
            'total_moves': self.move_count,
            'food_eaten': self.food_eaten,
            'strategies_used': strategies_used,
            'average_snake_length': sum(d['snake_length'] for d in self.decision_log) / len(self.decision_log) if self.decision_log else 0
        }

    def print_performance_report(self) -> None:
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

# Legacy stateless function for compatibility

def ai_move(game) -> None:
    """Choose next move for the game's snake (stateless, for legacy use)."""
    # Use the class-based AIController for logic
    ai = AIController()
    # Create a minimal GameState-like interface for legacy game
    class LegacyGameState:
        def __init__(self, game):
            self.grid_width = game.grid_width
            self.grid_height = game.grid_height
            self.snake = game.snake
            self.food = game.food
            self.grow = getattr(game, 'grow', False)
            self.direction = game.direction
        def get_snake_head(self):
            return self.snake[0]
        def get_snake_tail(self):
            return self.snake[-1]
        def get_snake_body(self):
            return list(self.snake)
        def is_valid_position(self, pos):
            x, y = pos
            return 0 <= x < self.grid_width and 0 <= y < self.grid_height
        def set_direction(self, direction, force=False):
            # Use the same logic as GameState for direction setting
            if not isinstance(direction, tuple) or len(direction) != 2:
                return
            ndx, ndy = direction
            cdx, cdy = self.direction
            if force or (ndx, ndy) != (-cdx, -cdy):
                game.direction = (ndx, ndy)
    legacy_state = LegacyGameState(game)
    ai.make_move(legacy_state) 