# AI Snake Game

A modern, well-architected Snake game built with Pygame featuring both manual and AI-controlled gameplay. The codebase has been refactored for improved testability, maintainability, and separation of concerns.

## 🏗️ Architecture

The project follows a clean architecture pattern with clear separation of concerns:

### Core Components

- **`models.py`**: Pure game state logic (no pygame dependencies)
- **`renderer.py`**: All rendering logic separated from game logic
- **`ai_controller.py`**: AI pathfinding and decision making
- **`game_controller.py`**: Game loop and coordination
- **`menu_controller.py`**: Menu navigation and logic
- **`config.py`**: Configuration constants and settings

### Key Improvements

✅ **Testable**: Game logic separated from pygame rendering  
✅ **Modular**: Each component has a single responsibility  
✅ **Maintainable**: Clear interfaces and dependency injection  
✅ **Extensible**: Easy to add new features or AI algorithms  

## 🚀 Quick Start

### Requirements
- Python 3.10 or later
- pygame 2.5.0

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd ai-snake

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Game

```bash
python main.py
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python tests/test_game_state.py
```

## 🎮 Game Features

- **Dual Gameplay**: Manual control or AI-controlled snake
- **Smart AI**: Pathfinding algorithm with fallback strategies
- **Customizable Settings**: Speed, grid size, NES-style mode
- **Score System**: High score tracking with multiplier bonuses
- **Pause Menu**: In-game settings and AI toggle

## 🎯 Controls

### Menu Navigation
- **Arrow Keys**: Navigate options
- **Enter**: Select option

### In-Game Controls
- **Arrow Keys**: Move snake (manual mode)
- **T**: Toggle AI control
- **Esc**: Pause menu
- **+/-**: Adjust game speed
- **R**: Restart game

## 🧪 Testing

The refactored architecture makes testing much easier:

```python
# Test game state logic without pygame
from snake.models import GameState

state = GameState(grid_width=10, grid_height=10)
state.move_snake()
assert state.get_snake_head() == (6, 5)  # Moved right
```

### Running Tests

```bash
# Install test dependencies
pip install pytest

# Run all tests
pytest tests/

# Run with coverage
pytest --cov=snake tests/
```

## 📁 Project Structure

```
ai-snake/
├── main.py                 # Entry point
├── requirements.txt        # Dependencies
├── .gitignore            # Git ignore rules
├── README.md             # This file
├── snake/                # Main package
│   ├── __init__.py
│   ├── config.py         # Configuration constants
│   ├── models.py         # Game state logic
│   ├── renderer.py       # Rendering logic
│   ├── ai_controller.py  # AI logic
│   ├── game_controller.py # Game coordination
│   ├── menu_controller.py # Menu logic
│   ├── game.py           # Legacy (deprecated)
│   ├── menu.py           # Legacy (deprecated)
│   └── ai.py             # Legacy (deprecated)
└── tests/                # Test suite
    └── test_game_state.py
```

## 🔧 Development

### Adding New Features

1. **Game Logic**: Add to `models.py` (pure logic)
2. **Rendering**: Add to `renderer.py` (visual only)
3. **AI**: Add to `ai_controller.py` (decision making)
4. **Menus**: Add to `menu_controller.py` (navigation)

### Code Style

- Follow PEP 8
- Use type hints
- Write docstrings for all public methods
- Keep components loosely coupled

## 🤖 AI Algorithm

The AI uses a sophisticated pathfinding approach:

1. **Primary Strategy**: BFS pathfinding to food
2. **Safety Check**: Ensures path to tail exists after move
3. **Fallback**: Open area calculation with wall distance
4. **Scoring**: Combines accessible area and wall proximity

## 📈 Performance

- **60 FPS**: Smooth gameplay at all speeds
- **Memory Efficient**: Minimal object creation
- **CPU Optimized**: Efficient pathfinding algorithms

## 🐛 Troubleshooting

### Common Issues

1. **Pygame not found**: Install with `pip install pygame==2.5.0`
2. **Display issues**: Ensure X11 forwarding in WSL
3. **Performance**: Lower speed setting if laggy

### Debug Mode

```python
# Add to main.py for debugging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.
