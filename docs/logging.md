# Logging Setup for AI Snake

This document describes the centralized logging system used in the AI Snake project, including configuration, usage, and examples for both human and coding agent (machine-readable) consumption.

---

## Centralized Logging Utility

All logging is managed via a utility in `src/utils/logging_utils.py`. This utility supports:
- Human-readable logs (console and file)
- Structured JSON logs for agent analysis
- Configurable output (file, console, log level, format)

### Setup

To initialize logging, call:

```python
from src.utils.logging_utils import setup_logging
setup_logging(log_to_file=True, log_to_console=True, log_level='INFO', json_mode=False)
```

- `log_to_file`: Write logs to a file (default: `logs/` directory)
- `log_to_console`: Output logs to the console
- `log_level`: Set the minimum log level (`DEBUG`, `INFO`, etc.)
- `json_mode`: If `True`, logs are output in JSON format (for agent consumption)

---

## Usage in Code

- **Always use named loggers**:
  ```python
  import logging
  logger = logging.getLogger(__name__)
  logger.info('This is an info message')
  logger.error('This is an error message')
  ```
- **Do not use `print()` for logging** (except for CLI prompts).
- **Test scripts** should also call `setup_logging` for consistent output.

---

## Human-Readable Log Example

```
2024-06-01 12:34:56,789 - INFO - ai.learning - EPISODE 10,Score=15,Length=120,Reward=100.0,Food=5,Deaths=1,Memory=500,Epsilon=0.45,DeathType=wall
2024-06-01 12:34:56,790 - INFO - ai.learning - Learning AI initialized with grid size (20, 20), device cuda, training=True
```

---

## JSON Log Example (for Agents)

```
{"event": "episode_end", "final_score": 15, "episode_length": 120, "total_reward": 100.0, "food_eaten": 5, "deaths": 1, "memory_size": 500, "epsilon": 0.45, "death_type": "wall"}
```

- Each episode end is logged as a JSON object for easy parsing by coding agents.
- Other key events (e.g., game over, model save/load) can also be logged in JSON.

---

## Best Practices

- Use `logger.info()` for general events, `logger.error()` for errors, and `logger.debug()` for detailed debugging.
- For agent analysis, always log structured data as JSON (see above).
- Avoid direct use of the root logger except in the logging setup.
- All modules should import and use their own named logger.

---

## Extending Logging

- To add new structured events, log a JSON object with an `event` key and relevant fields.
- To change log format, update `src/utils/logging_utils.py`.
- For new test scripts, always call `setup_logging` at the top.

---

## Troubleshooting

- If logs are missing, check that `setup_logging` is called before any logger usage.
- For import errors in test scripts, ensure you run tests from the project root or set `PYTHONPATH=src`.

---

For more details, see the code in `src/utils/logging_utils.py` and usage examples in `scripts/main.py` and `src/ai/learning.py`. 