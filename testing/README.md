# Testing & Benchmarking

This folder contains all test and benchmarking scripts for the AI Snake project.

## Structure
- `ai_performance_test.py` — Automated AI performance benchmarking
- `quick_ai_test.py` — Quick 10-game AI test
- `slow_ai_test.py` — Visual, slow AI test
- `test_ai_performance.py` — Compare trained vs untrained AI
- `test_ai_tracing.py` — Run a game with detailed AI tracing
- `test_game_state.py` — Game state unit tests
- `argparse_utils.py` — Shared argument parsing utilities
- `config.yaml` — Test-specific configuration
- `__pycache__/` — Python bytecode cache

## Usage
All scripts use the YAML config for test parameters. To run any script with the test config:

```bash
python <script_name>.py --config config.yaml
```

Or, to use the dedicated test config:

```bash
python <script_name>.py --config testing/config.yaml
```

## Example
Run a quick AI test:
```bash
python quick_ai_test.py --config config.yaml
```

Run a full performance benchmark:
```bash
python ai_performance_test.py --config config.yaml
```

## Customizing Tests
Edit `config.yaml` to change the number of games, speed, grid size, and other test parameters.

## Notes
- All imports now use `src.` for main code and local imports for utilities.
- This folder is self-contained for all testing and benchmarking needs. 