from ai_snake.config.loader import load_config, get_screen_size, CONFIG_FILE

# Game constants
config = load_config(CONFIG_FILE)
SCREEN_WIDTH, SCREEN_HEIGHT = get_screen_size(config)
BASE_SCREEN_SIZE = SCREEN_WIDTH
DEFAULT_GRID = (20, 20)

# Colors (no pygame dependency)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 30, 30)
GREEN = (30, 200, 30)
BLUE = (30, 30, 200)

HIGH_SCORE_FILE = 'high_score.txt'

# ---------------------------------------------------------------------------
# Fonts are created lazily so that merely importing this module does NOT call
# ``pygame.init()``.  This keeps headless mode and unit tests clean.
# ---------------------------------------------------------------------------
_FONT = None
_FONT_SMALL = None
_RETRO_FONT = None
_fonts_initialized = False


def _init_fonts():
    global _FONT, _FONT_SMALL, _RETRO_FONT, _fonts_initialized
    if _fonts_initialized:
        return
    import pygame
    if not pygame.get_init():
        pygame.init()
    _FONT = pygame.font.SysFont('Arial', 24)
    _FONT_SMALL = pygame.font.SysFont('Arial', 16)
    _RETRO_FONT = pygame.font.SysFont('Courier', 16, bold=True)
    _fonts_initialized = True


class _LazyFont:
    """Descriptor that initializes fonts on first attribute access."""
    def __init__(self, name):
        self._name = name

    def __getattr__(self, item):
        _init_fonts()
        font = globals()[self._name]
        return getattr(font, item)

    def __call__(self, *args, **kwargs):
        _init_fonts()
        return globals()[self._name](*args, **kwargs)


# Public font objects — behave like pygame.font.Font but delay init.
FONT = _LazyFont('_FONT')
FONT_SMALL = _LazyFont('_FONT_SMALL')
RETRO_FONT = _LazyFont('_RETRO_FONT')
