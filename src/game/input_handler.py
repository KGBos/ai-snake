import pygame

class InputHandler:
    def __init__(self, game_controller):
        self.game_controller = game_controller

    def handle_input(self, event: pygame.event.Event) -> bool:
        """Handle a single input event. Returns True if game should continue."""
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                return False  # Quit on Q key
            elif event.key == pygame.K_ESCAPE:
                return False  # Signal to pause
            elif event.key == pygame.K_t:
                self.game_controller.ai = not self.game_controller.ai
            elif event.key == pygame.K_l:
                if self.game_controller.learning_ai_controller:
                    self.game_controller.learning_ai = not self.game_controller.learning_ai
            elif event.key == pygame.K_m:
                if self.game_controller.learning_ai and self.game_controller.learning_ai_controller:
                    self.game_controller.manual_teaching_mode = not getattr(self.game_controller, 'manual_teaching_mode', False)
            elif event.key == pygame.K_p:
                if self.game_controller.learning_ai_controller:
                    self.game_controller.learning_ai_controller.set_training_mode(not self.game_controller.learning_ai_controller.training)
            elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                self.game_controller.speed += 1
            elif event.key == pygame.K_MINUS:
                self.game_controller.speed = max(1, self.game_controller.speed - 1)
            elif not self.game_controller.ai and not self.game_controller.learning_ai:
                self.handle_direction_input(event.key)
            elif self.game_controller.learning_ai and getattr(self.game_controller, 'manual_teaching_mode', False):
                self.handle_direction_input(event.key)
        return True

    def handle_direction_input(self, key: int):
        """Handle direction input for manual control."""
        old_direction = self.game_controller.game_state.direction
        if key == pygame.K_UP:
            self.game_controller.game_state.set_direction((0, -1))
        elif key == pygame.K_DOWN:
            self.game_controller.game_state.set_direction((0, 1))
        elif key == pygame.K_LEFT:
            self.game_controller.game_state.set_direction((-1, 0))
        elif key == pygame.K_RIGHT:
            self.game_controller.game_state.set_direction((1, 0)) 