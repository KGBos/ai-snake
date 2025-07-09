# FIXME: Review this file for potential issues or improvements
from abc import ABC, abstractmethod
from typing import Optional
from ai_snake.game.models import GameState

class BaseRenderer(ABC):
    @abstractmethod
    def render(self, game_state: GameState, current_time: int, info: Optional[dict] = None):
        pass

    @abstractmethod
    def clear_screen(self):
        pass 