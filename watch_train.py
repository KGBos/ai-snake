#!/usr/bin/env python3
"""
Watchable DQN Training: Runs many episodes in a single persistent Pygame window.
Lets you visually observe the agent learning in real time.
"""

import pygame
import argparse
import time
from snake.game_controller import GameController


def watch_train(episodes=100, speed=30, grid_size=(15, 15), model_path=None, save_interval=50):
    pygame.init()
    print(f"Starting watchable training for {episodes} episodes at speed {speed}.")
    print("Close the window or press ESC to stop early.")

    game_controller = GameController(
        speed=speed,
        ai=False,
        learning_ai=True,
        grid=grid_size,
        auto_advance=True,
        model_path=model_path
    )

    running = True
    episode = 0
    while running and episode < episodes:
        game_controller.reset()
        game_controller.episode_count = episode  # Ensure correct display
        print(f"Episode {episode+1}/{episodes}...")
        
        # Run one episode (returns False if user closes window)
        running = game_controller.run_game_loop()
        episode += 1

        # Save model at intervals
        if episode % save_interval == 0:
            game_controller.save_model(f"watchtrain_model_ep{episode}.pth")
            print(f"Model saved at episode {episode}")

        # Small delay for visibility (optional, can comment out)
        # time.sleep(0.2)

    # Final save
    game_controller.save_model("watchtrain_model_final.pth")
    print("Training complete! Model saved as 'watchtrain_model_final.pth'")
    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch DQN agent train visually in a single window.")
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes to run')
    parser.add_argument('--speed', type=int, default=30, help='Game speed (higher=faster)')
    parser.add_argument('--grid', type=int, nargs=2, default=[15, 15], help='Grid size (width height)')
    parser.add_argument('--model', type=str, default=None, help='Path to pre-trained model (optional)')
    parser.add_argument('--save-interval', type=int, default=50, help='How often to save the model')
    args = parser.parse_args()

    watch_train(
        episodes=args.episodes,
        speed=args.speed,
        grid_size=tuple(args.grid),
        model_path=args.model,
        save_interval=args.save_interval
    ) 