import pygame
import random
import os
from collections import deque

# Game constants
BASE_SCREEN_SIZE = 400
SCREEN_WIDTH = BASE_SCREEN_SIZE
SCREEN_HEIGHT = BASE_SCREEN_SIZE
DEFAULT_GRID = (20, 20)

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 30, 30)
GREEN = (30, 200, 30)
BLUE = (30, 30, 200)

pygame.init()
FONT = pygame.font.SysFont('Arial', 24)
FONT_SMALL = pygame.font.SysFont('Arial', 16)
RETRO_FONT = pygame.font.SysFont('Courier', 16, bold=True)
HIGH_SCORE_FILE = 'high_score.txt'


class SnakeGame:
    def __init__(self, speed=10, ai=False, grid=DEFAULT_GRID, nes_mode=False):
        self.speed = speed
        self.ai = ai
        self.grid_width, self.grid_height = grid
        self.cell_size = min(SCREEN_WIDTH // self.grid_width,
                             SCREEN_HEIGHT // self.grid_height)
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('AI Snake')
        self.font = RETRO_FONT if nes_mode else FONT
        self.nes_mode = nes_mode
        self.score = 0
        self.last_food_time = None
        self.high_score = self.load_high_score()
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
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                yield nx, ny

    def bfs(self):
        start = self.snake[0]
        goal = self.food  # cache food location
        queue = deque([start])
        came_from = {start: None}
        obstacles = set(list(self.snake)[:-1])  # exclude tail since it will move unless growing
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
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = hx + dx, hy + dy
            if not (0 <= nx < self.grid_width and 0 <= ny < self.grid_height):
                continue
            if (nx, ny) in list(self.snake)[:-1]:
                continue
            area = self.open_area((nx, ny))
            wall_dist = min(nx, self.grid_width-1-nx, ny, self.grid_height-1-ny)
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
        if key == pygame.K_UP and self.direction != (0,1):
            self.direction = (0,-1)
        elif key == pygame.K_DOWN and self.direction != (0,-1):
            self.direction = (0,1)
        elif key == pygame.K_LEFT and self.direction != (1,0):
            self.direction = (-1,0)
        elif key == pygame.K_RIGHT and self.direction != (-1,0):
            self.direction = (1,0)


def draw_centered_text(screen, text, y):
    surf = FONT.render(text, True, WHITE)
    rect = surf.get_rect(center=(SCREEN_WIDTH//2, y))
    screen.blit(surf, rect)


def settings_menu(settings):
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    selection = 0
    speeds = {0:5, 1:10, 2:15}
    grids = {3:(10,10), 4:(20,20), 5:(30,30)}
    while True:
        screen.fill(BLACK)
        draw_centered_text(screen, 'Settings', 80)
        option_texts = [
            'Speed: Slow',
            'Speed: Normal',
            'Speed: Fast',
            'Grid: 10x10',
            'Grid: 20x20',
            'Grid: 30x30',
            f'NES Style: {"On" if settings.get("nes", False) else "Off"}',
            'Back'
        ]
        for i, val in enumerate(option_texts):
            color = BLUE if selection == i else WHITE
            text = FONT.render(val, True, color)
            rect = text.get_rect(center=(SCREEN_WIDTH//2, 140 + i * 30))
            screen.blit(text, rect)
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return settings
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selection = (selection - 1) % len(option_texts)
                elif event.key == pygame.K_DOWN:
                    selection = (selection + 1) % len(option_texts)
                elif event.key == pygame.K_RETURN:
                    if selection in speeds:
                        settings['speed'] = speeds[selection]
                    elif selection in grids:
                        settings['grid'] = grids[selection]
                    elif selection == 6:
                        settings['nes'] = not settings.get('nes', False)
                    elif selection == 7:
                        return settings
        clock.tick(15)


def main_menu():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    options = ['Play Manual', 'Play AI', 'Settings', 'Quit']
    selection = 0
    settings = {'speed':10, 'grid':DEFAULT_GRID, 'nes':False}
    while True:
        screen.fill(BLACK)
        draw_centered_text(screen, 'AI Snake', 60)
        for i, opt in enumerate(options):
            color = BLUE if selection == i else WHITE
            text = FONT.render(opt, True, color)
            if selection == i:
                text = FONT.render(opt, True, color, BLACK)  # add background to selected
            rect = text.get_rect(center=(SCREEN_WIDTH//2, 120 + i * 40))
            screen.blit(text, rect)
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None, None
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selection = (selection - 1) % len(options)
                elif event.key == pygame.K_DOWN:
                    selection = (selection + 1) % len(options)
                elif event.key == pygame.K_RETURN:
                    if selection == 0:
                        return settings, False
                    elif selection == 1:
                        return settings, True
                    elif selection == 2:
                        settings = settings_menu(settings)
                    elif selection == 3:
                        return None, None
        clock.tick(15)


def pause_menu(game):
    selection = 0
    options = ['Resume', 'Toggle AI', 'Speed: Slow', 'Speed: Normal', 'Speed: Fast']
    while True:
        game.screen.fill(BLACK)
        draw_centered_text(game.screen, 'Paused', 60)
        for i, opt in enumerate(options):
            color = BLUE if selection == i else WHITE
            text = FONT.render(opt, True, color)
            rect = text.get_rect(center=(SCREEN_WIDTH//2, 120 + i * 40))
            game.screen.blit(text, rect)
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game.game_over = True
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selection = (selection - 1) % len(options)
                elif event.key == pygame.K_DOWN:
                    selection = (selection + 1) % len(options)
                elif event.key == pygame.K_RETURN:
                    if selection == 0:
                        return
                    elif selection == 1:
                        game.ai = not game.ai
                    elif selection in [2,3,4]:
                        speeds = {2:5, 3:10, 4:15}
                        game.speed = speeds[selection]
        game.clock.tick(15)


def main():
    settings = {'speed':10, 'grid':DEFAULT_GRID, 'nes':False}
    while True:
        result = main_menu()
        if result == (None, None):
            break
        settings, ai = result
        game = SnakeGame(speed=settings['speed'], ai=ai, grid=settings.get('grid', DEFAULT_GRID), nes_mode=settings.get('nes', False))
        while not game.game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game.game_over = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pause_menu(game)
                        continue
                    elif event.key == pygame.K_t:
                        game.ai = not game.ai
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                        game.speed += 1
                    elif event.key == pygame.K_MINUS:
                        game.speed = max(1, game.speed - 1)
                    elif not game.ai:
                        game.set_direction(event.key)
            game.update()
            game.draw()
            game.clock.tick(game.speed)
        game.show_game_over()
    pygame.quit()


if __name__ == '__main__':
    main()
