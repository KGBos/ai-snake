import os
import random
from collections import deque

import pygame

from .constants import (
    BLACK, BLUE, GREEN, RED, WHITE, SCREEN_WIDTH, SCREEN_HEIGHT,
    HIGH_SCORE_FILE
)


class SnakeGame:
    def __init__(self, speed=10, ai=False, grid=(20, 20), nes_mode=False):
        self.speed = speed
        self.ai = ai
        self.grid_width, self.grid_height = grid
        self.cell_size = min(SCREEN_WIDTH // self.grid_width,
                             SCREEN_HEIGHT // self.grid_height)
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('AI Snake')
        from .constants import FONT, RETRO_FONT
        self.font = RETRO_FONT if nes_mode else FONT
        self.nes_mode = nes_mode
        self.score = 0
        self.last_food_time = None
        self.high_score = self.load_high_score()
        self.clock = pygame.time.Clock()
        self.reset()

    def load_high_score(self):
        if os.path.exists(HIGH_SCORE_FILE):
            with open(HIGH_SCORE_FILE, 'r') as f:
                try:
                    return int(f.read().strip())
                except ValueError:
                    return 0
        return 0

    def save_high_score(self):
        with open(HIGH_SCORE_FILE, 'w') as f:
            f.write(str(self.high_score))

    def reset(self):
        self.direction = (1, 0)
        self.snake = deque([(self.grid_width // 2, self.grid_height // 2)])
        self.spawn_food()
        self.grow = False
        self.game_over = False
        self.score = 0
        self.last_food_time = None

    def spawn_food(self):
        if len(self.snake) >= self.grid_width * self.grid_height:
            self.game_over = True
            return
        while True:
            self.food = (
                random.randint(0, self.grid_width - 1),
                random.randint(0, self.grid_height - 1))
            if self.food not in self.snake:
                break

    def neighbors(self, pos):
        x, y = pos
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                yield nx, ny

    def bfs(self):
        start = self.snake[0]
        goal = self.food
        queue = deque([start])
        came_from = {start: None}
        obstacles = set(list(self.snake)[:-1])
        while queue:
            current = queue.popleft()
            if current == goal:
                break
            for nxt in self.neighbors(current):
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

    def open_area(self, start):
        queue = deque([start])
        visited = {start}
        obstacles = set(self.snake)
        count = 0
        while queue:
            x, y = queue.popleft()
            count += 1
            for nx, ny in self.neighbors((x, y)):
                if (nx, ny) in obstacles or (nx, ny) in visited:
                    continue
                visited.add((nx, ny))
                queue.append((nx, ny))
        return count

    def ai_move(self):
        hx, hy = self.snake[0]
        moves = []
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = hx + dx, hy + dy
            if not (0 <= nx < self.grid_width and 0 <= ny < self.grid_height):
                continue
            if (nx, ny) in list(self.snake)[:-1]:
                continue
            area = self.open_area((nx, ny))
            wall_dist = min(nx, self.grid_width - 1 - nx, ny, self.grid_height - 1 - ny)
            dist_food = abs(nx - self.food[0]) + abs(ny - self.food[1])
            score = area + wall_dist - dist_food * 0.1
            moves.append((score, (dx, dy)))
        if moves:
            moves.sort(reverse=True)
            self.direction = moves[0][1]

    def update(self):
        if self.ai:
            self.ai_move()
        self.move_snake()
        self.check_collision()
        self.handle_growth()

    def move_snake(self):
        hx, hy = self.snake[0]
        dx, dy = self.direction
        nx, ny = hx + dx, hy + dy
        self.snake.appendleft((nx, ny))

    def check_collision(self):
        hx, hy = self.snake[0]
        if (hx, hy) in list(self.snake)[1:] or not (0 <= hx < self.grid_width and 0 <= hy < self.grid_height):
            self.game_over = True
            return
        if (hx, hy) == self.food:
            self.grow = True
            now = pygame.time.get_ticks()
            if self.last_food_time and now - self.last_food_time <= 3000:
                self.score += 2
            else:
                self.score += 1
            self.last_food_time = now
            if self.score > self.high_score:
                self.high_score = self.score
            self.spawn_food()

    def handle_growth(self):
        if not self.grow:
            self.snake.pop()
        else:
            self.grow = False

    def draw_cell(self, pos, color):
        x, y = pos
        rect = pygame.Rect(x * self.cell_size, y * self.cell_size,
                           self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, color, rect, border_radius=4)
        if self.nes_mode:
            pygame.draw.rect(self.screen, WHITE, rect, 1)

    def draw(self):
        from .constants import FONT_SMALL
        self.screen.fill(BLACK)
        for i, segment in enumerate(self.snake):
            self.draw_cell(segment, BLUE if i == 0 else GREEN)
        self.draw_cell(self.food, RED)
        if self.nes_mode:
            for x in range(self.grid_width):
                pygame.draw.line(self.screen, WHITE, (x * self.cell_size, 0), (x * self.cell_size, SCREEN_HEIGHT), 1)
            for y in range(self.grid_height):
                pygame.draw.line(self.screen, WHITE, (0, y * self.cell_size), (SCREEN_WIDTH, y * self.cell_size), 1)
        multiplier = 2 if self.last_food_time and pygame.time.get_ticks() - self.last_food_time <= 3000 else 1
        score_text = self.font.render(f'Score: {self.score} x{multiplier}', True, WHITE)
        self.screen.blit(score_text, (5, 5))
        help_lines = [
            'Arrow keys: Move',
            'T: Toggle AI',
            'Esc: Pause',
            '+/-: Speed',
            'R: Restart'
        ]
        for i, line in enumerate(help_lines):
            help_surf = FONT_SMALL.render(line, True, WHITE)
            self.screen.blit(help_surf, (5, SCREEN_HEIGHT - (len(help_lines) - i) * 18))
        pygame.display.flip()

    def game_loop(self):
        while not self.game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.game_over = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.game_over = True
                    elif event.key == pygame.K_t:
                        self.ai = not self.ai
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                        self.speed += 1
                    elif event.key == pygame.K_MINUS:
                        self.speed = max(1, self.speed - 1)
                    elif not self.ai:
                        self.set_direction(event.key)
            self.update()
            self.draw()
            self.clock.tick(self.speed)
        self.show_game_over()

    def show_game_over(self):
        from .constants import FONT
        if self.score > self.high_score:
            self.high_score = self.score
            self.save_high_score()
        self.screen.fill(BLACK)
        over_text = self.font.render('Game Over', True, WHITE)
        score_text = self.font.render(f'Score: {self.score}', True, WHITE)
        high_text = self.font.render(f'High: {self.high_score}', True, WHITE)
        prompt = self.font.render('Press Enter to return to Main Menu', True, WHITE)
        self.screen.blit(over_text, over_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 40)))
        self.screen.blit(score_text, score_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2)))
        self.screen.blit(high_text, high_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 40)))
        self.screen.blit(prompt, prompt.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 80)))
        pygame.display.flip()
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                    waiting = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    self.reset()
                    self.game_loop()
                    return
            self.clock.tick(5)

    def set_direction(self, key):
        if key == pygame.K_UP and self.direction != (0, 1):
            self.direction = (0, -1)
        elif key == pygame.K_DOWN and self.direction != (0, -1):
            self.direction = (0, 1)
        elif key == pygame.K_LEFT and self.direction != (1, 0):
            self.direction = (-1, 0)
        elif key == pygame.K_RIGHT and self.direction != (-1, 0):
            self.direction = (1, 0)
