# AI Snake Game

This is a simple snake game written with Pygame. You can choose to play manually or let a basic AI control the snake. A main menu allows you to start the game, toggle AI, and adjust speed settings. The code is now split into a small `snake` package containing the game logic and menu helpers.

## Requirements
- Python 3.12 or later
- pygame 2.5.0

Install dependencies with:

```bash
pip install pygame==2.5.0
```

## Running

Run the game with:

```bash
python3 main.py
```

Use the arrow keys to navigate the menu. During the game, press `Esc` to open the pause menu. From there you can toggle AI control or change the speed.

Press `t` at any time during the game to quickly toggle AI control on or off.

## Custom Assets

The `assets/` folder contains two subdirectories:

* `fonts/` – place `nes_font.ttf` here if you want to enable the NES style score display.
* `sprites/` – add 16x16 PNG images for the snake and background tiles using the names described in `assets/sprites/README.md`.

Placeholder files are included so the game runs even if you have not provided custom assets.
