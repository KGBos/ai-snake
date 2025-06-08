import pygame

from snake.game import SnakeGame
from snake.menu import main_menu, pause_menu
from snake.config import DEFAULT_GRID

settings = {'speed': 10, 'grid': DEFAULT_GRID, 'nes': False}
game = SnakeGame(speed=settings['speed'], ai=False,
                 grid=settings.get('grid', DEFAULT_GRID),
                 nes_mode=settings.get('nes', False))

selection = 0  # Assuming this is defined somewhere for menu navigation

# Example of corrected indentation for menu navigation
if selection == 0:
    pass  # Do nothing
elif selection == 1:
    game.ai = not game.ai
elif selection in [2, 3, 4]:
    speeds = {2: 5, 3: 10, 4: 15}
    game.speed = speeds[selection]

game.clock.tick(15)


def main():
    settings = {'speed': 10, 'grid': DEFAULT_GRID, 'nes': False}
    while True:
        result = main_menu()
        if result == (None, None):
            break
        settings, ai = result
        game = SnakeGame(speed=settings['speed'], ai=ai, grid=settings.get('grid', DEFAULT_GRID), nes_mode=settings.get('nes', False))
        while not game.game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game.game_over = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pause_menu(game)
                        continue
                    elif event.key == pygame.K_t:
                        game.ai = not game.ai
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                        game.speed += 1
                    elif event.key == pygame.K_MINUS:
                        game.speed = max(1, game.speed - 1)
                    elif not game.ai:
                        game.set_direction(event.key)
            game.update()
            game.draw()
            game.clock.tick(game.speed)
        game.show_game_over()
    pygame.quit()


if __name__ == '__main__':
    main()
