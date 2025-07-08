import os
import random
from collections import deque

import pygame

from config.config import (
    BLACK, BLUE, GREEN, RED, WHITE, SCREEN_WIDTH, SCREEN_HEIGHT,
    HIGH_SCORE_FILE
)
from config.loader import *
from game.models import GameState


class SnakeGame:
    def __init__(self, speed=10, ai=False, grid=(20, 20), nes_mode=False):
        self.speed = speed
        self.ai = ai
        self.grid_width, self.grid_height = grid
        self.cell_size = min(SCREEN_WIDTH // self.grid_width,
                             SCREEN_HEIGHT // self.grid_height)
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('AI Snake')
        from config.config import FONT, RETRO_FONT
        self.font = RETRO_FONT if nes_mode else FONT
        self.nes_mode = nes_mode
        self.score = 0
        self.last_food_time = None
        self.high_score = self.load_high_score()
        self.clock = pygame.time.Clock()
        self.reset()

    def load_high_score(self):
        try:
            if os.path.exists(HIGH_SCORE_FILE):
                with open(HIGH_SCORE_FILE, 'r') as f:
                    try:
                        return int(f.read().strip())
                    except ValueError:
                        return 0
        except (OSError, IOError):
            return 0
        return 0

    def save_high_score(self):
        try:
            with open(HIGH_SCORE_FILE, 'w') as f:
                f.write(str(self.high_score))
        except (OSError, IOError):
            pass

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
            from ai.rule_based import ai_move
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

    def draw_cell(self, pos, color):
        x, y = pos
        rect = pygame.Rect(x * self.cell_size, y * self.cell_size,
                           self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, color, rect, border_radius=4)
        if self.nes_mode:
            pygame.draw.rect(self.screen, WHITE, rect, 1)

    def draw(self):
        from config.config import FONT_SMALL
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
        from config.config import FONT
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
        while True:
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                        return
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        self.reset()
                        self.game_loop()
                        waiting = False
                        break
                self.clock.tick(5)
            # After a restart, break out of the outer loop to avoid recursion
            break

    def set_direction(self, key):
        # Use set_direction to ensure robust 180-degree turn prevention
        if key == pygame.K_UP:
            self.set_direction_internal((0, -1))
        elif key == pygame.K_DOWN:
            self.set_direction_internal((0, 1))
        elif key == pygame.K_LEFT:
            self.set_direction_internal((-1, 0))
        elif key == pygame.K_RIGHT:
            self.set_direction_internal((1, 0))

    def set_direction_internal(self, new_direction):
        # Use the same logic as GameState for direction setting
        if not isinstance(new_direction, tuple) or len(new_direction) != 2:
            return
        ndx, ndy = new_direction
        cdx, cdy = self.direction
        if (ndx, ndy) != (-cdx, -cdy):
            self.direction = (ndx, ndy)
