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
    SAFE_PATH_LOOKAHEAD_NORMAL = 4
    SAFE_PATH_LOOKAHEAD_ENDGAME = 7
    ENDGAME_LENGTH_RATIO = 0.7
    
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
        if start == goal:
            return []
            
        queue = deque([start])
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        
        while queue:
            current = queue.popleft()
            if current == goal:
                break
            
            for nxt in self.get_neighbors(current, width, height):
                if nxt in obstacles and nxt != goal:
                    continue
                if nxt in came_from:
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
        """Check whether a path exists from start to goal using BFS."""
        if start == goal:
            return True
            
        queue = deque([start])
        visited = {start}
        
        while queue:
            current = queue.popleft()
            if current == goal:
                return True
            
            for key_neighbor in self.get_neighbors(current, width, height):
                if key_neighbor in obstacles and key_neighbor != goal:
                    continue
                if key_neighbor in visited:
                    continue
                visited.add(key_neighbor)
                queue.append(key_neighbor)
        return False

    def calculate_open_area(self, start: Tuple[int, int], obstacles: Set[Tuple[int, int]], 
                            width: int, height: int) -> int:
        """Return number of reachable cells from start."""
        queue = deque([start])
        visited = {start}
        count = 0
        while queue:
            current = queue.popleft()
            count += 1
            for nxt in self.get_neighbors(current, width, height):
                if nxt in obstacles or nxt in visited:
                    continue
                visited.add(nxt)
                queue.append(nxt)
        return count

    def is_path_safe(self, game_state: GameState, path: List[Tuple[int, int]], lookahead: int = 5) -> bool:
        """Simulate following the path for 'lookahead' steps and check if tail remains reachable."""
        if not path:
            return False
            
        simulated_state = deepcopy(game_state)
        steps_to_check = min(lookahead, len(path))
        
        for step in range(steps_to_check):
            next_cell = path[step]
            
            # Simulate move
            simulated_state.snake.appendleft(next_cell)
            if next_cell != simulated_state.food and not simulated_state.grow:
                simulated_state.snake.pop()
            simulated_state.grow = False
            
            # Check if we can reach tail from new head
            # Note: body[:-1] excludes tail because valid move can go to current tail position
            # providing we didn't just grow. But to be safe and simple, let's treat body as obstacles.
            # Ideally, we should simulate the exact snake movement.
            # In the original code: set(simulated_state.get_snake_body()[:-1])
            obstacles = set(simulated_state.get_snake_body())
            # The tail will move, so it shouldn't be an obstacle for the NEXT move, 
            # but here we are checking if a path EXISTS from head to tail.
            # If we just moved, the old tail is gone (unless we grew).
            # The 'tail' target is the NEW tail.
            
            head = simulated_state.get_snake_head()
            tail = simulated_state.get_snake_tail()
            
            # For pathfinding to tail, the tail strictly isn't an obstacle, it's the target.
            # But the body excluding the tail IS an obstacle.
            obstacles.discard(tail)
            
            if not self.path_exists(head, tail, obstacles, 
                                  simulated_state.grid_width, simulated_state.grid_height):
                return False
                
        return True

    def is_creating_box(self, game_state: GameState, next_cell: Tuple[int, int]) -> bool:
        """Detect if moving to next_cell would create a tight box/loop (3 sides blocked)."""
        x, y = next_cell
        blocked = 0
        snake_body_set = set(game_state.get_snake_body())
        
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = x+dx, y+dy
            if not (0 <= nx < game_state.grid_width and 0 <= ny < game_state.grid_height):
                blocked += 1
            elif (nx, ny) in snake_body_set:
                blocked += 1
        return blocked >= 3

    def get_path_diversity(self, paths: List[List[Tuple[int, int]]]) -> List[List[Tuple[int, int]]]:
        """Return a list of paths with maximal diversity (spread out moves)."""
        if not paths:
            return []
        # For now, shuffle to avoid always picking the same path
        random.shuffle(paths)
        return paths

    def _log_decision(self, info: Dict[str, Any], level: str = 'info') -> None:
        self.decision_log.append(info)
        if self.enable_tracing:
            msg = f"Move {self.move_count}: {info.get('strategy', 'Unknown')} - {info.get('reasoning', [])} - Direction: {info.get('direction')}"
            if level == 'warning':
                self.logger.warning(msg)
            else:
                self.logger.info(msg)

    def _get_direction_from_path(self, head: Tuple[int, int], path: List[Tuple[int, int]]) -> Tuple[int, int]:
        next_cell = path[0]
        return (next_cell[0] - head[0], next_cell[1] - head[1])

    def _try_safe_path_to_food(self, game_state, head, food, obstacles, grid_w, grid_h, endgame, 
                             decision_info) -> Optional[Tuple[int, int]]:
        path = self.bfs_path(head, food, obstacles, grid_w, grid_h)
        if not path:
            return None

        # Lookahead safety check
        lookahead = self.SAFE_PATH_LOOKAHEAD_ENDGAME if endgame else self.SAFE_PATH_LOOKAHEAD_NORMAL
        if self.is_path_safe(game_state, path, lookahead=lookahead):
            next_cell = path[0]
            if not self.is_creating_box(game_state, next_cell):
                direction = self._get_direction_from_path(head, path)
                decision_info['strategy'] = 'path_to_food_safe'
                decision_info['reasoning'].append(f"Safe path to food: {path[:3]}...")
                decision_info['direction'] = direction
                self._log_decision(decision_info)
                return direction
        return None

    def _try_risky_path_to_food(self, game_state, head, food, obstacles, grid_w, grid_h, 
                              starvation, in_loop, decision_info) -> Optional[Tuple[int, int]]:
        if not (starvation or in_loop):
            return None
            
        path = self.bfs_path(head, food, obstacles, grid_w, grid_h)
        if path:
            direction = self._get_direction_from_path(head, path)
            reason = f"Starved {self.moves_since_food} moves" if starvation else "Loop detected"
            decision_info['strategy'] = 'forced_risky_food'
            decision_info['reasoning'].append(f"{reason}, taking risky path: {path[:3]}...")
            decision_info['direction'] = direction
            self._log_decision(decision_info, level='warning')
            return direction
        return None

    def _try_tail_chasing(self, game_state, head, tail, obstacles, grid_w, grid_h, decision_info) -> Optional[Tuple[int, int]]:
        tail_path = self.bfs_path(head, tail, obstacles, grid_w, grid_h)
        if tail_path:
            next_cell = tail_path[0]
            if not self.is_creating_box(game_state, next_cell):
                direction = self._get_direction_from_path(head, tail_path)
                decision_info['strategy'] = 'tail_chase'
                decision_info['reasoning'].append(f"Chasing tail: {tail_path[:3]}...")
                decision_info['direction'] = direction
                self._log_decision(decision_info)
                return direction
        return None

    def _try_open_area_fallback(self, game_state, head, snake_body, obstacles, grid_w, grid_h, endgame, decision_info) -> Optional[Tuple[int, int]]:
        best_score = None
        best_dir = None
        
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = head[0] + dx, head[1] + dy
            if not game_state.is_valid_position((nx, ny)):
                continue
            if (nx, ny) in obstacles:
                continue
            
            # Avoid immediate death traps
            if self.is_creating_box(game_state, (nx, ny)):
                continue
                
            area = self.calculate_open_area((nx, ny), set(snake_body), grid_w, grid_h)
            wall_dist = min(nx, grid_w - 1 - nx, ny, grid_h - 1 - ny)
            
            # Heuristic score
            score = area * (2.0 if endgame else 1.0) + wall_dist * self.WALL_DIST_WEIGHT
            
            if best_score is None or score > best_score:
                best_score = score
                best_dir = (dx, dy)
        
        if best_dir:
            decision_info['strategy'] = 'open_area_fallback'
            decision_info['reasoning'].append("Using open area strategy")
            decision_info['direction'] = best_dir
            self._log_decision(decision_info)
            return best_dir
        return None

    def _try_any_valid_move(self, game_state, head, obstacles, decision_info) -> Optional[Tuple[int, int]]:
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = head[0] + dx, head[1] + dy
            if game_state.is_valid_position((nx, ny)) and (nx, ny) not in obstacles:
                decision_info['strategy'] = 'last_resort_any_valid_move'
                decision_info['reasoning'].append("Taking any available move.")
                decision_info['direction'] = (dx, dy)
                self._log_decision(decision_info, level='warning')
                return (dx, dy)
        return None

    def get_best_direction(self, game_state: GameState) -> Optional[Tuple[int, int]]:
        head = game_state.get_snake_head()
        tail = game_state.get_snake_tail()
        snake_body = game_state.get_snake_body()
        obstacles = set(list(snake_body)[:-1]) # Tail is a valid move target usually
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

        # Analysis
        endgame = len(snake_body) > (grid_w * grid_h) * self.ENDGAME_LENGTH_RATIO
        starvation = self.moves_since_food >= self.STARVATION_LIMIT
        
        # Loop detection
        self.loop_history.append(head)
        if len(self.loop_history) > self.LOOP_HISTORY_SIZE:
            self.loop_history.pop(0)
        head_counts = {pos: self.loop_history.count(pos) for pos in set(self.loop_history)}
        in_loop = any(count >= self.LOOP_REPEAT_THRESHOLD for count in head_counts.values())

        # Strategy 1: Safe Path to Food
        direction = self._try_safe_path_to_food(game_state, head, food, obstacles, grid_w, grid_h, endgame, decision_info)
        if direction: return direction

        # Strategy 2: Risky Path to Food (if starving or stuck)
        direction = self._try_risky_path_to_food(game_state, head, food, obstacles, grid_w, grid_h, starvation, in_loop, decision_info)
        if direction: return direction

        # Strategy 3: Chase Tail
        direction = self._try_tail_chasing(game_state, head, tail, obstacles, grid_w, grid_h, decision_info)
        if direction: return direction
        
        # Strategy 4: Open Area Fallback
        direction = self._try_open_area_fallback(game_state, head, snake_body, obstacles, grid_w, grid_h, endgame, decision_info)
        if direction: return direction
        
        # Strategy 5: Any Valid Move
        direction = self._try_any_valid_move(game_state, head, obstacles, decision_info)
        if direction: return direction

        # No valid move
        decision_info['strategy'] = 'no_valid_move'
        decision_info['reasoning'].append("No valid move found!")
        self._log_decision(decision_info, level='warning')
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