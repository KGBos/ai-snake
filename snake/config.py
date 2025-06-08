import os
import pygame

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

# Asset directories
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')
SPRITES_DIR = os.path.join(ASSETS_DIR, 'sprites')
FONTS_DIR = os.path.join(ASSETS_DIR, 'fonts')
NES_FONT_PATH = os.path.join(FONTS_DIR, 'nes_font.ttf')

# NES style font. Fallback to a system monospace font if the TTF is missing.
if os.path.exists(NES_FONT_PATH):
    RETRO_FONT = pygame.font.Font(NES_FONT_PATH, 16)
else:
    RETRO_FONT = pygame.font.SysFont('Courier', 16, bold=True)

HIGH_SCORE_FILE = 'high_score.txt'
