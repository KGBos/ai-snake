# AI Snake Game

A Python-based Snake game with both rule-based and deep learning AI. The project features a modular, testable architecture, comprehensive benchmarking, and real-time training visualization.

## Features

- **Classic Snake Gameplay**: Eat food, grow, avoid collisions
- **Rule-based AI**: Pathfinding (A* and BFS) with fallback strategies
- **Deep Q-Learning AI**: Neural network that learns to play Snake
- **GPU Acceleration**: CUDA support for fast training
- **Modular Design**: Clean separation of game logic, AI, rendering, and menus
- **Comprehensive Testing**: Automated test suite and benchmarking scripts
- **Performance Monitoring**: AI tracing, statistics, and batch testing
- **Training Visualization**: Real-time plots and statistics

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ai-snake
   ```
2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Play the Game
```bash
python main.py
```

### Run with Rule-based AI
```bash
python main.py
# Press 'T' to toggle AI on/off
```

### Run with Learning AI (DQN)
```bash
python main.py --learning --model <model_path>
# Press 'L' to toggle learning AI
```

### Train the Learning AI
```bash
python train_ai.py --episodes 1000
# Advanced:
python train_ai.py --episodes 5000 --grid-size 20 20 --device cuda
```

## Game Controls

- **Arrow Keys**: Move snake (manual mode)
- **T**: Toggle rule-based AI
- **L**: Toggle learning AI (when available)
- **S**: Save learning model
- **+/-**: Adjust game speed
- **ESC**: Pause/Quit
- **ENTER**: Continue after game over
- **R**: Restart game

## AI Modes

### Rule-based AI
- **A* Pathfinding**: Finds optimal path to food
- **Fallback**: Open area and wall avoidance if blocked
- **Tracing**: Detailed decision logs (enable in code or with key)

### Deep Q-Learning AI
- **Neural Network**: 3 conv layers, 2 FC layers, 4 outputs (actions)
- **Experience Replay**: Learns from past moves
- **Epsilon-Greedy**: Balances exploration/exploitation
- **GPU/CPU**: CUDA or CPU selectable
- **Model Saving/Loading**: Save with 'S', load with --model

## Training the Learning AI

### Basic Training
```bash
python train_ai.py --episodes 1000
```

### Advanced Training Options
```bash
python train_ai.py --episodes 2000 --grid-size 15 15
python train_ai.py --episodes 1000 --device cpu
python train_ai.py --episodes 1000 --model existing_model.pth
python train_ai.py --episodes 5000 --save-interval 200
```

### Training Output
- **Model files**: `model_episode_X.pth`
- **Statistics**: `stats_episode_X.json`
- **Plots**: `training_plots_episode_X.png`
- **Output directory**: `training_output_YYYYMMDD_HHMMSS/`

## Testing & Benchmarking

All test scripts are in the `testing/` folder. Use the YAML config for test parameters.

### Run All Tests
```bash
pytest
```

### Run Specific Tests
```bash
pytest tests/test_models.py
pytest tests/test_ai_controller.py
```

### AI Performance Testing
```bash
python testing/ai_performance_test.py --config testing/config.yaml
python testing/quick_ai_test.py --config testing/config.yaml
python testing/test_ai_performance.py --model <model_path> --episodes 50
```

- **ai_performance_test.py**: Batch test AI, detailed stats
- **quick_ai_test.py**: Fast 10-game test
- **test_ai_performance.py**: Compare trained vs untrained AI
- **test_ai_tracing.py**: Run with detailed AI tracing
- **test_game_state.py**: Game state unit tests

Edit `testing/config.yaml` to change test parameters (games, speed, grid size, etc).

## Performance Monitoring

- **AI Tracing**: Enable to see AI's decision process
- **Training Stats**: Rewards, food, epsilon, etc. during training
- **Batch Testing**: Analyze average score, food, length, and strategy usage

## Configuration

- **Grid Size**: Default 20x20, customizable
- **Speed**: Adjustable
- **NES Mode**: Retro style
- **Auto-advance**: Skip game over for batch runs
- **AI Settings**: Learning rate, gamma, epsilon, memory size, batch size

## Architecture

### Core Components
- `src/models.py`: Game state and logic
- `src/renderer.py`: Graphics/UI
- `src/ai_controller.py`: Rule-based AI
- `src/learning_ai_controller.py`: DQN learning AI
- `src/dqn_agent.py`: Neural network and training logic
- `src/game_controller.py`: Game loop
- `src/menu_controller.py`: Menus
- `train_ai.py`: Training script
- `main.py`: Entry point
- `testing/`: Test and benchmarking scripts

### Neural Network
- **Input**: 3-channel grid (snake, food, head)
- **Conv Layers**: 3×3 kernels, 3 layers
- **FC Layers**: 256, 128 neurons
- **Output**: 4 actions (up, down, left, right)

## Project Structure
```
ai-snake/
├── src/                  # Core game modules
│   ├── models.py
│   ├── renderer.py
│   ├── ai_controller.py
│   ├── learning_ai_controller.py
│   ├── dqn_agent.py
│   ├── game_controller.py
│   └── menu_controller.py
├── testing/              # Test & benchmarking scripts
│   ├── ai_performance_test.py
│   ├── quick_ai_test.py
│   ├── test_ai_performance.py
│   ├── test_ai_tracing.py
│   ├── test_game_state.py
│   └── config.yaml
├── train_ai.py           # Training script
├── main.py               # Game entry point
├── requirements.txt      # Dependencies
└── docs/                 # Documentation
```

## Troubleshooting

- **CUDA not available**: `python train_ai.py --device cpu`
- **Training slow**: Reduce grid size, use GPU, increase speed
- **Model not learning**: Increase episodes, check reward function, monitor epsilon

## License

Open source. See LICENSE for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit a pull request

## Version History

See [CHANGELOG.md](CHANGELOG.md) for details.
