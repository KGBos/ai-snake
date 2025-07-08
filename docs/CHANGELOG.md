# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2024-12-19

### Added
- **Deep Q-Learning AI System**: Complete neural network-based learning AI
  - Convolutional neural network for state representation
  - Experience replay buffer for stable training
  - Epsilon-greedy exploration strategy
  - GPU acceleration with CUDA support
  - Target network for stable Q-learning updates

- **DQN Agent Implementation** (`snake/dqn_agent.py`):
  - `SnakeDQN` neural network class with 3 conv layers + 2 FC layers
  - `DQNAgent` class with full DQN training pipeline
  - State representation using 3-channel grid (snake body, food, head)
  - Experience replay with configurable memory size
  - Automatic epsilon decay for exploration/exploitation balance

- **Learning AI Controller** (`snake/learning_ai_controller.py`):
  - `LearningAIController` for integrating DQN with game loop
  - `RewardCalculator` for custom reward functions
  - Real-time training statistics display
  - Model saving/loading functionality

- **Training Infrastructure**:
  - `train_ai.py` script for automated training sessions
  - Progress tracking with tqdm progress bars
  - Automatic model and statistics saving
  - Training visualization with matplotlib plots
  - Configurable training parameters (episodes, grid size, device)

- **Enhanced Game Controller**:
  - Learning AI mode integration
  - Real-time training statistics display
  - Model saving during gameplay (S key)
  - Learning AI toggle (L key)
  - Auto-advance support for training

- **Command Line Interface**:
  - `--learning` flag for learning AI mode
  - `--model` parameter for loading pre-trained models
  - `--episodes` for training duration
  - `--auto-advance` for continuous training
  - `--grid` for custom grid sizes

- **Dependencies**:
  - PyTorch for neural network implementation
  - NumPy for numerical operations
  - Matplotlib for training visualization
  - tqdm for progress tracking

### Changed
- **Updated `main.py`**: Added command line argument parsing for learning mode
- **Enhanced `game_controller.py`**: Integrated learning AI with existing game loop
- **Updated `requirements.txt`**: Added deep learning dependencies
- **Comprehensive README**: Complete documentation for learning system

### Technical Details
- **Neural Network Architecture**:
  - Input: 3×20×20 grid representation
  - Conv1: 3→32 channels, 3×3 kernel
  - Conv2: 32→64 channels, 3×3 kernel  
  - Conv3: 64→64 channels, 3×3 kernel
  - FC1: 25600→256 neurons
  - FC2: 256→128 neurons
  - Output: 128→4 action values

- **Training Parameters**:
  - Learning rate: 0.001
  - Gamma (discount factor): 0.99
  - Epsilon: 1.0 → 0.01 (decay: 0.995)
  - Memory size: 10,000 experiences
  - Batch size: 32
  - Target network update: every 100 steps

- **Reward Function**:
  - +10.0 for eating food
  - -10.0 for dying
  - -0.1 per move (efficiency penalty)
  - +0.1 for survival

## [1.1.0] - 2024-12-18

### Added
- **Automated Batch Testing**: Comprehensive AI performance testing system
  - `test_ai_performance.py` for large-scale testing (100+ games)
  - `quick_test.py` for rapid testing (10-50 games)
  - Statistical analysis with averages, medians, and distributions
  - Performance categorization (high/medium/low performers)
  - Strategy usage analysis across multiple games

- **Enhanced AI Tracing**: Detailed decision-making analysis
  - Real-time strategy logging for each move
  - Performance metrics including survival rate and food efficiency
  - Strategy analysis showing algorithm usage patterns
  - Detailed reasoning for each AI decision

- **Auto-advance Feature**: Skip game over screen for continuous testing
  - Integrated into game controller for seamless batch testing
  - Configurable through command line arguments
  - Maintains high score tracking during batch runs

### Changed
- **Fixed Food Eaten Tracking**: Corrected timing of collision and food checks
- **Updated Score Tracking**: Separated food eaten count from score multiplier
- **Enhanced Performance Reporting**: More detailed statistics and categorization

### Technical Details
- **Batch Testing Features**:
  - Configurable game count (10-1000+ games)
  - Adjustable speed for faster testing
  - Detailed statistical analysis
  - Performance categorization system
  - Strategy usage tracking

- **Sample Output**:
  ```
  === AI PERFORMANCE ANALYSIS ===
  Games completed: 50
  
  📊 SCORE STATISTICS:
    Average score: 18.7
    Median score: 17.0
    Best score: 35
    Worst score: 3
    Standard deviation: 8.2
  
  🐍 SNAKE LENGTH STATISTICS:
    Average final length: 19.7
    Median final length: 18.0
    Longest snake: 36
    Shortest snake: 4
  
  🧠 STRATEGY USAGE:
    path_to_food: 1247 times (89.2%)
    open_area_fallback: 151 times (10.8%)
  
  🏆 PERFORMANCE CATEGORIES:
    High performers (20+ score): 15 games (30.0%)
    Medium performers (10-19 score): 25 games (50.0%)
    Low performers (<10 score): 10 games (20.0%)
  ```

## [1.0.0] - 2024-12-17

### Added
- **Complete Modular Architecture**: Refactored codebase for better maintainability
  - `models.py`: Pure game state logic (no pygame dependencies)
  - `renderer.py`: All rendering logic separated from game logic
  - `ai_controller.py`: AI pathfinding and decision making
  - `game_controller.py`: Game loop and coordination
  - `menu_controller.py`: Menu navigation and logic
  - `config.py`: Configuration constants and settings

- **Comprehensive Test Suite**: Full testing infrastructure
  - `tests/test_models.py`: Game state logic tests
  - `tests/test_ai_controller.py`: AI algorithm tests
  - Automated test runner with pytest
  - High test coverage for core game logic

- **Enhanced AI System**: Intelligent pathfinding with fallback strategies
  - A* pathfinding algorithm for optimal food collection
  - Fallback strategy for wall avoidance
  - Performance tracking and metrics
  - Detailed AI tracing for debugging

- **Advanced Menu System**: Feature-rich menu interface
  - Settings configuration (speed, grid size, NES mode)
  - AI tracing toggle
  - High score tracking
  - Responsive menu navigation

- **Performance Monitoring**: Detailed AI analysis tools
  - Real-time AI decision logging
  - Performance metrics and statistics
  - Strategy usage analysis
  - Automated performance testing

### Changed
- **Architecture Overhaul**: Complete separation of concerns
  - Game logic independent of rendering
  - AI logic separated from game state
  - Menu system modular and extensible
  - Configuration centralized

- **Enhanced Documentation**: Comprehensive README and documentation
  - Detailed installation instructions
  - Architecture overview
  - Development guidelines
  - Troubleshooting guide

### Technical Details
- **AI Algorithm**:
  - Primary: BFS pathfinding to food
  - Safety: Path to tail verification
  - Fallback: Open area calculation
  - Scoring: Area + wall proximity

- **Testing Infrastructure**:
  - Pure game logic tests (no pygame)
  - AI algorithm validation
  - Performance benchmarking
  - Automated test suite

## [0.1.0] - 2024-12-16

### Added
- **Initial Project Setup**: Basic Snake game implementation
- **Pygame Integration**: Graphics and input handling
- **Basic Game Logic**: Snake movement, food collection, collision detection
- **Simple AI**: Basic pathfinding implementation
- **Menu System**: Basic game menu
- **High Score Tracking**: Persistent score storage

### Technical Details
- **Core Features**:
  - Snake movement and growth
  - Food spawning and collection
  - Wall and self collision detection
  - Score system with multipliers
  - Basic AI pathfinding
  - Menu navigation

- **File Structure**:
  - `main.py`: Entry point
  - `snake/game.py`: Main game logic
  - `snake/menu.py`: Menu system
  - `snake/ai.py`: AI implementation
  - `requirements.txt`: Dependencies 