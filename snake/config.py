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
RETRO_FONT = pygame.font.SysFont('Courier', 16, bold=True)

HIGH_SCORE_FILE = 'high_score.txt'
