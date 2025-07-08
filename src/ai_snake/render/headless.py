from ai_snake.render.base import BaseRenderer
from ai_snake.game.models import GameState
from typing import Optional

class HeadlessRenderer(BaseRenderer):
    def render(self, game_state: GameState, current_time: int, info: Optional[dict] = None):
        pass  # No-op for headless mode

    def clear_screen(self):
        pass  # No-op for headless mode 