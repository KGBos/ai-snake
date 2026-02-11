import pygame
from ai_snake.render.renderer import GameRenderer
from ai_snake.config.config import SCREEN_WIDTH, SCREEN_HEIGHT

class VideoRenderer(GameRenderer):
    """Renderer specifically for capturing video frames off-screen."""
    
    def __init__(self, nes_mode: bool = False):
        # Initialize parent with headless=False so it sets up fonts etc usually
        # But we don't want it to create a display surface with set_mode
        # so we trick it or override __init__.
        # Easiest is to call super with headless=True to skip set_mode,
        # then manually create the surface.
        super().__init__(nes_mode=nes_mode, headless=True)
        
        # Create an offscreen surface
        self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        
        # Re-enable drawing (headless=True usually disables drawing in _draw_frame)
        self.headless = False 

    def get_frame(self):
        """Return the current frame as a numpy array (transposed for correct orientation)."""
        # Pygame uses (W, H, C), we might need it differently for Video
        # But usually we return what surfarray gives.
        # Note: surfarray.array3d returns (W, H, 3).
        # We need to transpose to (H, W, 3) for standard image processing/Wandb
        return pygame.surfarray.array3d(self.screen).transpose(1, 0, 2)
