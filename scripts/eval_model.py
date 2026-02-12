#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime, timezone
from statistics import mean, pstdev
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

# Ensure imports work when running from source checkout (package lives under ../src).
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_root = os.path.join(repo_root, "src")
sys.path.insert(0, src_root)

from ai_snake.ai.dqn import DQNAgent
from ai_snake.config.loader import get_grid_size, load_config
from ai_snake.game.models import GameState


ACTION_TO_DIRECTION: Dict[int, Tuple[int, int]] = {
    0: (0, -1),
    1: (0, 1),
    2: (-1, 0),
    3: (1, 0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one or more DQN Snake models with fixed seeds and export CSV/JSON stats."
    )
    parser.add_argument(
        "--model",
        nargs="+",
        required=True,
        help="One or more model checkpoint paths to evaluate.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes to run per model.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed. Each episode uses seed + episode_index.",
    )
    parser.add_argument(
        "--grid",
        type=int,
        nargs=2,
        default=None,
        metavar=("WIDTH", "HEIGHT"),
        help="Grid size override. Defaults to values from --config.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Config path used for defaults when flags are omitted.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Step cap per episode.",
    )
    parser.add_argument(
        "--starvation-threshold",
        type=int,
        default=50,
        help="Moves without score increase before marking starvation death.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default=None,
        help="Torch device override (default: auto-detect).",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Output file prefix (without extension). Defaults to outputs/eval_<timestamp>.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_device(device_override: Optional[str]) -> str:
    if device_override:
        return device_override
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def run_episode(
    agent: DQNAgent,
    grid: Tuple[int, int],
    max_steps: int,
    starvation_threshold: int,
) -> Dict[str, Union[int, str]]:
    state = GameState(grid_width=grid[0], grid_height=grid[1])
    steps = 0
    steps_without_food = 0
    prev_score = state.score

    while not state.game_over and steps < max_steps:
        action = agent.get_action(state, training=False)
        direction = ACTION_TO_DIRECTION[action]
        state.set_direction(direction, force=True)
        state.move_snake()
        state.check_collision(current_time=0)
        steps += 1

        if state.score > prev_score:
            steps_without_food = 0
        else:
            steps_without_food += 1

        if (not state.game_over) and steps_without_food >= starvation_threshold:
            state.set_starvation_death()

        prev_score = state.score

    if not state.game_over and steps >= max_steps:
        state.set_death_type("step_limit")

    return {
        "score": int(state.score),
        "steps": steps,
        "death_type": state.death_type or "unknown",
        "fruits_eaten": max(0, len(state.snake) - 1),
    }


def summarize(episodes: List[Dict[str, Union[int, str]]]) -> Dict[str, Union[float, int, Dict[str, int]]]:
    scores = [int(e["score"]) for e in episodes]
    steps = [int(e["steps"]) for e in episodes]
    fruits = [int(e["fruits_eaten"]) for e in episodes]
    deaths: Dict[str, int] = {}
    for ep in episodes:
        death = str(ep["death_type"])
        deaths[death] = deaths.get(death, 0) + 1

    return {
        "episodes": len(episodes),
        "score_mean": mean(scores) if scores else 0.0,
        "score_std": pstdev(scores) if len(scores) > 1 else 0.0,
        "score_min": min(scores) if scores else 0,
        "score_max": max(scores) if scores else 0,
        "steps_mean": mean(steps) if steps else 0.0,
        "steps_std": pstdev(steps) if len(steps) > 1 else 0.0,
        "fruits_mean": mean(fruits) if fruits else 0.0,
        "death_type_counts": deaths,
    }


def ensure_output_prefix(output_prefix: Optional[str]) -> str:
    if output_prefix:
        out = output_prefix
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out = os.path.join("outputs", f"eval_{timestamp}")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    return out


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    grid = tuple(args.grid) if args.grid else get_grid_size(config)
    device = detect_device(args.device)
    output_prefix = ensure_output_prefix(args.output_prefix)

    all_rows: List[Dict[str, Union[int, str]]] = []
    model_summaries: Dict[str, Dict[str, Union[float, int, Dict[str, int]]]] = {}

    for model_path in args.model:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        agent = DQNAgent(device=device)
        agent.load_model(model_path)

        episodes: List[Dict[str, Union[int, str]]] = []
        for episode_index in range(args.episodes):
            set_seed(args.seed + episode_index)
            episode = run_episode(
                agent=agent,
                grid=grid,
                max_steps=args.max_steps,
                starvation_threshold=args.starvation_threshold,
            )
            row = {
                "model": model_path,
                "episode": episode_index + 1,
                "seed": args.seed + episode_index,
                **episode,
            }
            episodes.append(episode)
            all_rows.append(row)

        model_summaries[model_path] = summarize(episodes)

    csv_path = f"{output_prefix}.csv"
    json_path = f"{output_prefix}.json"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "episode", "seed", "score", "fruits_eaten", "steps", "death_type"],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    payload = {
        "meta": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "episodes_per_model": args.episodes,
            "seed_base": args.seed,
            "grid": {"width": grid[0], "height": grid[1]},
            "max_steps": args.max_steps,
            "starvation_threshold": args.starvation_threshold,
            "device": device,
        },
        "summary_by_model": model_summaries,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Evaluation complete for {len(args.model)} model(s).")
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")

    for model_path, summary in model_summaries.items():
        print(
            f"- {model_path}: "
            f"score_mean={summary['score_mean']:.3f}, "
            f"score_std={summary['score_std']:.3f}, "
            f"score_max={summary['score_max']}, "
            f"steps_mean={summary['steps_mean']:.2f}"
        )


if __name__ == "__main__":
    main()
