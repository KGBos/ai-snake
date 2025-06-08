import pygame
import random
from collections import deque

# Game constants
CELL_SIZE = 20
GRID_WIDTH = 20
GRID_HEIGHT = 20
SCREEN_WIDTH = CELL_SIZE * GRID_WIDTH
SCREEN_HEIGHT = CELL_SIZE * GRID_HEIGHT

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 30, 30)
GREEN = (30, 200, 30)
BLUE = (30, 30, 200)

pygame.init()
FONT = pygame.font.SysFont('Arial', 24)


class SnakeGame:
    def __init__(self, speed=10, ai=False):
        self.speed = speed
        self.ai = ai
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('AI Snake')
        self.reset()

    def reset(self):
        self.direction = (1, 0)
        self.snake = deque([(GRID_WIDTH // 2, GRID_HEIGHT // 2)])
        self.spawn_food()
        self.grow = False
        self.game_over = False

    def spawn_food(self):
        while True:
            self.food = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
            if self.food not in self.snake:
                break

    def neighbors(self, pos):
        x, y = pos
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT:
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

    def ai_move(self):
        path = self.bfs()
        if path:
            nx, ny = path[0]
            hx, hy = self.snake[0]
            self.direction = (nx - hx, ny - hy)
        else:
            # fall back to safe random move
            possible = []
            hx, hy = self.snake[0]
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                nx, ny = hx + dx, hy + dy
                if (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT and
                        (nx, ny) not in list(self.snake)[:-1]):
                    possible.append((dx, dy))
            if possible:
                self.direction = random.choice(possible)

    def update(self):
        if self.ai:
            self.ai_move()
        hx, hy = self.snake[0]
        dx, dy = self.direction
        nx, ny = hx + dx, hy + dy
        if (nx, ny) in self.snake or not (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT):
            self.game_over = True
            return
        self.snake.appendleft((nx, ny))
        if (nx, ny) == self.food:
            self.grow = True
            self.spawn_food()
        if not self.grow:
            self.snake.pop()
        else:
            self.grow = False

    def draw_cell(self, pos, color):
        x, y = pos
        rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(self.screen, color, rect)

    def draw(self):
        self.screen.fill(BLACK)
        for segment in self.snake:
            self.draw_cell(segment, GREEN)
        self.draw_cell(self.food, RED)
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
                    elif not self.ai:
                        if event.key == pygame.K_UP and self.direction != (0,1):
                            self.direction = (0,-1)
                        elif event.key == pygame.K_DOWN and self.direction != (0,-1):
                            self.direction = (0,1)
                        elif event.key == pygame.K_LEFT and self.direction != (1,0):
                            self.direction = (-1,0)
                        elif event.key == pygame.K_RIGHT and self.direction != (-1,0):
                            self.direction = (1,0)
            self.update()
            self.draw()
            self.clock.tick(self.speed)
        self.show_game_over()

    def show_game_over(self):
        text = FONT.render('Game Over - Press Enter', True, WHITE)
        rect = text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2))
        self.screen.blit(text, rect)
        pygame.display.flip()
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                    waiting = False
            self.clock.tick(5)


def draw_centered_text(screen, text, y):
    surf = FONT.render(text, True, WHITE)
    rect = surf.get_rect(center=(SCREEN_WIDTH//2, y))
    screen.blit(surf, rect)


def settings_menu(settings):
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    selection = 0
    speeds = [5, 10, 15]
    while True:
        screen.fill(BLACK)
        draw_centered_text(screen, 'Settings', 80)
        for i, val in enumerate(['Slow', 'Normal', 'Fast']):
            color = BLUE if selection == i else WHITE
            text = FONT.render(val, True, color)
            rect = text.get_rect(center=(SCREEN_WIDTH//2, 140 + i * 40))
            screen.blit(text, rect)
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return settings
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selection = (selection - 1) % 3
                elif event.key == pygame.K_DOWN:
                    selection = (selection + 1) % 3
                elif event.key == pygame.K_RETURN:
                    settings['speed'] = speeds[selection]
                    return settings
        clock.tick(15)


def main_menu():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    options = ['Play Manual', 'Play AI', 'Settings', 'Quit']
    selection = 0
    while True:
        screen.fill(BLACK)
        draw_centered_text(screen, 'AI Snake', 60)
        for i, opt in enumerate(options):
            color = BLUE if selection == i else WHITE
            text = FONT.render(opt, True, color)
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
                        return {'speed':10}, False
                    elif selection == 1:
                        return {'speed':10}, True
                    elif selection == 2:
                        settings = {'speed':10}
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
    settings = {'speed':10}
    while True:
        result = main_menu()
        if result == (None, None):
            break
        settings, ai = result
        game = SnakeGame(speed=settings['speed'], ai=ai)
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
                    elif not game.ai:
                        if event.key == pygame.K_UP and game.direction != (0,1):
                            game.direction = (0,-1)
                        elif event.key == pygame.K_DOWN and game.direction != (0,-1):
                            game.direction = (0,1)
                        elif event.key == pygame.K_LEFT and game.direction != (1,0):
                            game.direction = (-1,0)
                        elif event.key == pygame.K_RIGHT and game.direction != (-1,0):
                            game.direction = (1,0)
            game.update()
            game.draw()
            game.clock.tick(game.speed)
        game.show_game_over()
    pygame.quit()


if __name__ == '__main__':
    main()
