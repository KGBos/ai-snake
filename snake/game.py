import os
import random
from collections import deque

import pygame

from .config import (
    BLACK, BLUE, GREEN, RED, WHITE, SCREEN_WIDTH, SCREEN_HEIGHT,
    HIGH_SCORE_FILE, SPRITES_DIR
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
        from .config import FONT, RETRO_FONT
        self.font = RETRO_FONT if nes_mode else FONT
        self.nes_mode = nes_mode
        if self.nes_mode:
            self.load_assets()
        self.score = 0
        self.last_food_time = None
        self.high_score = self.load_high_score()
        self.clock = pygame.time.Clock()
        self.reset()

    def load_assets(self):
        """Load NES style sprites and create CRT overlay."""
        def load(name, color):
            """Load sprite image or create a colored square if missing."""
            path = os.path.join(SPRITES_DIR, name)
            if os.path.exists(path):
                surf = pygame.image.load(path).convert_alpha()
            else:
                surf = pygame.Surface((self.cell_size, self.cell_size))
                surf.fill(color)
            return pygame.transform.scale(surf, (self.cell_size, self.cell_size))

        self.sprite_head = load('snake_head.png', WHITE)
        self.sprite_body = load('snake_body.png', GREEN)
        turn_path = os.path.join(SPRITES_DIR, 'snake_turn.png')
        self.sprite_turn = load('snake_turn.png', GREEN) if os.path.exists(turn_path) else self.sprite_body
        self.sprite_tail = load('snake_tail.png', BLUE)
        self.sprite_food = load('food.png', RED)
        self.bg_tile = load('background_tile.png', BLACK)
        self.border_tile = load('border_tile.png', WHITE)

        # Precompute CRT overlay surface for bonus effect
        self.crt_overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        for y in range(0, SCREEN_HEIGHT, 2):
            pygame.draw.line(self.crt_overlay, (0, 0, 0, 40), (0, y), (SCREEN_WIDTH, y))

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


    def update(self):
        if self.ai:
            from .ai import ai_move
            ai_move(self)
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

    def dir_to_angle(self, d):
        mapping = {(1, 0): 0, (0, 1): 90, (-1, 0): 180, (0, -1): 270}
        return mapping.get(d, 0)

    def blit_sprite(self, sprite, pos, angle=0):
        rotated = pygame.transform.rotate(sprite, angle)
        self.screen.blit(rotated, (pos[0] * self.cell_size, pos[1] * self.cell_size))

    def draw_cell(self, pos, color):
        x, y = pos
        rect = pygame.Rect(x * self.cell_size, y * self.cell_size,
                           self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, color, rect, border_radius=4)
        if self.nes_mode:
            pygame.draw.rect(self.screen, WHITE, rect, 1)

    def draw(self):
        from .config import FONT_SMALL
        if self.nes_mode:
            # draw tiled background
            for x in range(self.grid_width):
                for y in range(self.grid_height):
                    self.blit_sprite(self.bg_tile, (x, y))
            # optional border
            for x in range(-1, self.grid_width + 1):
                self.blit_sprite(self.border_tile, (x, -1))
                self.blit_sprite(self.border_tile, (x, self.grid_height))
            for y in range(self.grid_height):
                self.blit_sprite(self.border_tile, (-1, y))
                self.blit_sprite(self.border_tile, (self.grid_width, y))

            # draw snake with sprites
            for i, segment in enumerate(self.snake):
                if i == 0:
                    angle = self.dir_to_angle(self.direction)
                    self.blit_sprite(self.sprite_head, segment, angle)
                elif i == len(self.snake) - 1:
                    prev = self.snake[i - 1]
                    d = (segment[0] - prev[0], segment[1] - prev[1])
                    self.blit_sprite(self.sprite_tail, segment, self.dir_to_angle(d))
                else:
                    prev = self.snake[i - 1]
                    nxt = self.snake[i + 1]
                    d1 = (segment[0] - prev[0], segment[1] - prev[1])
                    d2 = (nxt[0] - segment[0], nxt[1] - segment[1])
                    if d1[0] == d2[0] or d1[1] == d2[1]:
                        angle = 0 if d1[0] != 0 else 90
                        self.blit_sprite(self.sprite_body, segment, angle if d1[0] >= 0 or d1[1] >= 0 else angle + 180)
                    else:
                        key = {(1,0):(0,1), (0,1):(-1,0), (-1,0):(0,-1), (0,-1):(1,0)}
                        angle = {((1,0),(0,1)):0, ((0,1),(-1,0)):90, ((-1,0),(0,-1)):180, ((0,-1),(1,0)):270}.get((d1,d2))
                        if angle is None:
                            angle = {((0,1),(1,0)):270, ((-1,0),(0,1)):0, ((0,-1),(-1,0)):90, ((1,0),(0,-1)):180}.get((d1,d2),0)
                        self.blit_sprite(self.sprite_turn, segment, angle)

            # draw food
            self.blit_sprite(self.sprite_food, self.food)

            # score display
            score_text = self.font.render(f'SCORE: {self.score:06d}', True, WHITE)
            self.screen.blit(score_text, (5, 5))

            # help text
            for i, line in enumerate(['Arrow keys: Move', 'T: Toggle AI', 'Esc: Pause', '+/-: Speed', 'R: Restart']):
                help_surf = FONT_SMALL.render(line, True, WHITE)
                self.screen.blit(help_surf, (5, SCREEN_HEIGHT - (5 - i) * 18))

            # apply CRT overlay
            self.screen.blit(self.crt_overlay, (0, 0))
        else:
            self.screen.fill(BLACK)
            for i, segment in enumerate(self.snake):
                self.draw_cell(segment, BLUE if i == 0 else GREEN)
            self.draw_cell(self.food, RED)
            multiplier = 2 if self.last_food_time and pygame.time.get_ticks() - self.last_food_time <= 3000 else 1
            score_text = self.font.render(f'Score: {self.score} x{multiplier}', True, WHITE)
            self.screen.blit(score_text, (5, 5))
            for i, line in enumerate(['Arrow keys: Move', 'T: Toggle AI', 'Esc: Pause', '+/-: Speed', 'R: Restart']):
                help_surf = FONT_SMALL.render(line, True, WHITE)
                self.screen.blit(help_surf, (5, SCREEN_HEIGHT - (5 - i) * 18))

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
        from .config import FONT
        if self.score > self.high_score:
            self.high_score = self.score
            self.save_high_score()
        self.screen.fill(BLACK)
        over_text = self.font.render('Game Over', True, WHITE)
        score_text = self.font.render(f'SCORE: {self.score:06d}', True, WHITE)
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
