# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2024-12-19

### Added
- Initial changelog creation
- Comprehensive test suite for `GameState` class
- Modular architecture with separate controllers and renderers
- Development environment setup with virtual environment and dependencies
- Type hints throughout the codebase for better maintainability

### Changed
- Major refactor of the codebase for improved modularity and testability:
  - Separated game logic, rendering, AI, and menu logic into distinct modules
  - Introduced `GameState` for pure game logic, independent of rendering or Pygame
  - Added `GameController`, `MenuController`, `AIController`, and `Renderer` classes for clear separation of concerns
  - Updated `main.py` to use the new architecture
- Added comprehensive `.gitignore` and updated `requirements.txt` for development and testing
- Rewritten `README.md` to reflect the new architecture, usage, and development practices

### Deprecated
- Legacy modules (`game.py`, `menu.py`, `ai.py`) in favor of new architecture

### Removed
- Tight coupling between game logic and rendering
- Mixed responsibilities in the original `SnakeGame` class

### Fixed
- Improved testability by separating game logic from pygame dependencies
- Enhanced maintainability through clear module boundaries
- Better error handling in file operations for high score persistence

### Security
- No security changes in this release 