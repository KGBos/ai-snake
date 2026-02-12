import os
import sys

# Ensure imports work when running from source checkout (package lives under ../src).
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_root = os.path.join(repo_root, "src")
if src_root not in sys.path:
    sys.path.insert(0, src_root)

from ai_snake.cli import main


if __name__ == "__main__":
    main()
