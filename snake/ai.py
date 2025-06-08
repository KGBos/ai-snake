from collections import deque


def neighbors(pos, width, height):
    x, y = pos
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height:
            yield nx, ny


def bfs(start, goal, obstacles, width, height):
    """Breadth-first search returning path from start to goal."""
    queue = deque([start])
    came_from = {start: None}
    while queue:
        current = queue.popleft()
        if current == goal:
            break
        for nxt in neighbors(current, width, height):
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


def path_exists(start, goal, obstacles, width, height):
    """Check whether a path exists from start to goal."""
    queue = deque([start])
    visited = {start}
    while queue:
        current = queue.popleft()
        if current == goal:
            return True
        for nxt in neighbors(current, width, height):
            if nxt in obstacles or nxt in visited:
                continue
            visited.add(nxt)
            queue.append(nxt)
    return False


def open_area(start, obstacles, width, height):
    """Return number of reachable cells from start."""
    queue = deque([start])
    visited = {start}
    count = 0
    while queue:
        x, y = queue.popleft()
        count += 1
        for nx, ny in neighbors((x, y), width, height):
            if (nx, ny) in obstacles or (nx, ny) in visited:
                continue
            visited.add((nx, ny))
            queue.append((nx, ny))
    return count


def ai_move(game):
    """Choose next move for the game's snake."""
    head = game.snake[0]
    tail = game.snake[-1]
    obstacles = set(list(game.snake)[:-1])

    path = bfs(head, game.food, obstacles, game.grid_width, game.grid_height)
    if path:
        next_cell = path[0]
        future_snake = list(game.snake)
        future_snake.insert(0, next_cell)
        if next_cell != game.food and not game.grow:
            future_snake.pop()
        new_head = next_cell
        new_tail = future_snake[-1]
        if path_exists(new_head, new_tail, set(future_snake[:-1]), game.grid_width, game.grid_height):
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
        area = open_area((nx, ny), set(game.snake), game.grid_width, game.grid_height)
        wall_dist = min(nx, game.grid_width - 1 - nx, ny, game.grid_height - 1 - ny)
        score = area + wall_dist * 0.5
        if best_score is None or score > best_score:
            best_score = score
            best_dir = (dx, dy)
    if best_dir:
        game.direction = best_dir
