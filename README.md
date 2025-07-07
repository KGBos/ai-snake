# AI Snake Game

A Python-based Snake game with both rule-based and learning AI implementations. The game features a modular architecture with separate components for game logic, AI, rendering, and menus.

## Features

- **Classic Snake Gameplay**: Traditional snake game with food collection and collision detection
- **Rule-based AI**: Intelligent pathfinding AI using A* algorithm and fallback strategies
- **Deep Q-Learning AI**: Neural network-based AI that learns from experience
- **GPU Acceleration**: CUDA support for faster training on NVIDIA GPUs
- **Modular Architecture**: Clean separation of concerns with testable components
- **Comprehensive Testing**: Automated test suite for game logic
- **Performance Monitoring**: Detailed AI tracing and performance metrics
- **Training Visualization**: Real-time plots and statistics during training

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

### Train the Learning AI
```bash
# Quick training (1000 episodes)
python train_ai.py --episodes 1000

# Extended training with custom settings
python train_ai.py --episodes 5000 --grid-size 20 20 --device cuda
```

### Run with Pre-trained Model
```bash
python main.py --learning --model snake_dqn_model.pth
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
The traditional AI uses intelligent pathfinding:
- **A* Pathfinding**: Finds optimal path to food
- **Fallback Strategy**: Avoids walls and self when pathfinding fails
- **Performance Tracking**: Detailed metrics and tracing

### Deep Q-Learning AI
The learning AI uses reinforcement learning:
- **Neural Network**: Convolutional neural network for state representation
- **Experience Replay**: Stores and learns from past experiences
- **Epsilon-Greedy**: Balances exploration and exploitation
- **GPU Acceleration**: CUDA support for faster training

## Training the Learning AI

### Basic Training
```bash
python train_ai.py --episodes 1000
```

### Advanced Training Options
```bash
# Custom grid size
python train_ai.py --episodes 2000 --grid-size 15 15

# Use CPU instead of GPU
python train_ai.py --episodes 1000 --device cpu

# Load pre-trained model
python train_ai.py --episodes 1000 --model existing_model.pth

# Custom save interval
python train_ai.py --episodes 5000 --save-interval 200
```

### Training Output
The training script creates:
- **Model files**: `model_episode_X.pth`
- **Statistics**: `stats_episode_X.json`
- **Plots**: `training_plots_episode_X.png`
- **Output directory**: `training_output_YYYYMMDD_HHMMSS/`

## Architecture

### Core Components
- **`models.py`**: Game state and logic
- **`renderer.py`**: Graphics and UI rendering
- **`ai_controller.py`**: Rule-based AI implementation
- **`learning_ai_controller.py`**: DQN learning AI
- **`dqn_agent.py`**: Neural network and training logic
- **`game_controller.py`**: Main game loop coordination
- **`menu_controller.py`**: Menu system

### Neural Network Architecture
- **Input**: 3-channel grid representation (snake body, food, head)
- **Convolutional Layers**: 3 conv layers with ReLU activation
- **Fully Connected**: 2 FC layers with 256 and 128 neurons
- **Output**: 4 action values (up, down, left, right)

## Testing

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
# Batch test rule-based AI
python test_ai_performance.py --games 100 --speed 50

# Quick test
python quick_test.py --games 10
```

## Performance Monitoring

### AI Tracing
Enable detailed AI tracing to see decision-making:
```bash
python main.py
# Press 'T' to enable AI, then watch console output
```

### Training Statistics
During training, monitor:
- **Episode Rewards**: Total reward per episode
- **Average Reward**: Moving average over 100 episodes
- **Food Eaten**: Number of food items collected
- **Epsilon**: Exploration rate (decays over time)

## Configuration

### Game Settings
- **Grid Size**: Default 20x20, customizable
- **Speed**: Adjustable game speed
- **NES Mode**: Retro font and styling
- **Auto-advance**: Skip game over screen

### AI Settings
- **Learning Rate**: 0.001 (DQN)
- **Gamma**: 0.99 (discount factor)
- **Epsilon**: 1.0 → 0.01 (exploration decay)
- **Memory Size**: 10,000 experiences
- **Batch Size**: 32 for training

## Development

### Project Structure
```
ai-snake/
├── snake/                 # Core game modules
│   ├── models.py         # Game state
│   ├── renderer.py       # Graphics
│   ├── ai_controller.py  # Rule-based AI
│   ├── learning_ai_controller.py  # Learning AI
│   ├── dqn_agent.py     # Neural network
│   ├── game_controller.py # Game loop
│   └── menu_controller.py # Menus
├── tests/                # Test suite
├── train_ai.py          # Training script
├── test_ai_performance.py # Performance testing
├── main.py              # Entry point
└── requirements.txt     # Dependencies
```

### Adding Features
1. **New AI Strategy**: Add to `ai_controller.py`
2. **UI Changes**: Modify `renderer.py`
3. **Game Logic**: Update `models.py`
4. **Training**: Extend `dqn_agent.py`

## Troubleshooting

### Common Issues

**CUDA not available**:
```bash
python train_ai.py --device cpu
```

**Training too slow**:
- Reduce grid size: `--grid-size 15 15`
- Use GPU: `--device cuda`
- Increase speed: `--speed 20`

**Model not learning**:
- Increase episodes: `--episodes 5000`
- Check reward function in `learning_ai_controller.py`
- Monitor epsilon decay

### Performance Tips
- **GPU Training**: Use CUDA for 10x speedup
- **Batch Size**: Increase for faster training (if memory allows)
- **Grid Size**: Smaller grids train faster
- **Save Interval**: Balance between progress tracking and disk usage

## License

This project is open source. See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Version History

See [CHANGELOG.md](CHANGELOG.md) for detailed version history and release notes.
