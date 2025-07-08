import pygame

from .config import (
    BLUE, WHITE, BLACK, SCREEN_WIDTH, SCREEN_HEIGHT, DEFAULT_GRID,
    FONT, FONT_SMALL
)


def draw_centered_text(screen, text, y):
    surf = FONT.render(text, True, WHITE)
    rect = surf.get_rect(center=(SCREEN_WIDTH // 2, y))
    screen.blit(surf, rect)


def settings_menu(settings):
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    selection = 0
    speeds = {0: 5, 1: 10, 2: 15}
    grids = {3: (10, 10), 4: (20, 20), 5: (30, 30)}
    while True:
        screen.fill(BLACK)
        draw_centered_text(screen, 'Settings', 80)
        option_texts = [
            'Speed: Slow',
            'Speed: Normal',
            'Speed: Fast',
            'Grid: 10x10',
            'Grid: 20x20',
            'Grid: 30x30',
            f'NES Style: {"On" if settings.get("nes", False) else "Off"}',
            'Back'
        ]
        for i, val in enumerate(option_texts):
            color = BLUE if selection == i else WHITE
            text = FONT.render(val, True, color)
            rect = text.get_rect(center=(SCREEN_WIDTH // 2, 140 + i * 30))
            screen.blit(text, rect)
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return settings
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selection = (selection - 1) % len(option_texts)
                elif event.key == pygame.K_DOWN:
                    selection = (selection + 1) % len(option_texts)
                elif event.key == pygame.K_RETURN:
                    if selection in speeds:
                        settings['speed'] = speeds[selection]
                    elif selection in grids:
                        settings['grid'] = grids[selection]
                    elif selection == 6:
                        settings['nes'] = not settings.get('nes', False)
                    elif selection == 7:
                        return settings
        clock.tick(15)


def main_menu():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    options = ['AI Game', 'Human Game', 'Settings', 'Quit']
    selection = 0
    settings = {'speed': 10, 'grid': DEFAULT_GRID, 'nes': False}
    while True:
        screen.fill(BLACK)
        draw_centered_text(screen, 'AI Snake', 60)
        for i, opt in enumerate(options):
            color = BLUE if selection == i else WHITE
            text = FONT.render(opt, True, color)
            if selection == i:
                text = FONT.render(opt, True, color, BLACK)
            rect = text.get_rect(center=(SCREEN_WIDTH // 2, 120 + i * 40))
            screen.blit(text, rect)
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None, None
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selection = (selection - 1) % len(options)
                elif event.key == pygame.K_DOWN:
                    selection = (selection + 1) % len(options)
                elif event.key == pygame.K_RETURN:
                    if selection == 0:
                        return settings, True
                    elif selection == 1:
                        return settings, False
                    elif selection == 2:
                        settings = settings_menu(settings)
                    elif selection == 3:
                        return None, None
        clock.tick(15)


def pause_menu(game):
    selection = 0
    options = ['Resume', 'Toggle AI', 'Speed: Slow', 'Speed: Normal', 'Speed: Fast']
    while True:
        game.screen.fill(BLACK)
        draw_centered_text(game.screen, 'Paused', 60)
        for i, opt in enumerate(options):
            color = BLUE if selection == i else WHITE
            text = FONT.render(opt, True, color)
            rect = text.get_rect(center=(SCREEN_WIDTH // 2, 120 + i * 40))
            game.screen.blit(text, rect)
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game.game_over = True
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selection = (selection - 1) % len(options)
                elif event.key == pygame.K_DOWN:
                    selection = (selection + 1) % len(options)
                elif event.key == pygame.K_RETURN:
                    if selection == 0:
                        return
                    elif selection == 1:
                        game.ai = not game.ai
                    elif selection in [2, 3, 4]:
                        speeds = {2: 5, 3: 10, 4: 15}
                        game.speed = speeds[selection]
        game.clock.tick(15)
