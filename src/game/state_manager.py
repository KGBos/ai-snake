from game.models import GameState
import pygame

class GameStateManager:
    def __init__(self, grid_width, grid_height):
        self.game_state = GameState(grid_width=grid_width, grid_height=grid_height)

    def move_snake(self):
        self.game_state.move_snake()

    def check_collision(self, current_time):
        self.game_state.check_collision(current_time)

    def reset(self):
        self.game_state.reset()

    def get_snake_head(self):
        return self.game_state.get_snake_head()

    def get_snake_body(self):
        return self.game_state.get_snake_body()

    def set_direction(self, direction, force=False):
        self.game_state.set_direction(direction, force=force)

    def get_snake_tail(self):
        return self.game_state.get_snake_tail()

    def get_food(self):
        return self.game_state.food

    def get_direction(self):
        return self.game_state.direction

    def get_score(self):
        return self.game_state.score

    def is_game_over(self):
        return self.game_state.game_over

    def set_grow(self, value: bool):
        self.game_state.grow = value

    # Add more wrappers as needed for other GameState methods/attributes 