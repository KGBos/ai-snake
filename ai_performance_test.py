#!/usr/bin/env python3
"""
Automated AI performance testing script.
Runs multiple games at high speed to collect comprehensive performance data.
"""

import sys
import os
import time
import statistics
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from snake.game_controller import GameController
from snake.config import DEFAULT_GRID


class AIPerformanceTester:
    """Automated tester for AI performance across multiple games."""
    
    def __init__(self, num_games: int = 100, speed: int = 60, grid_size: tuple = (20, 20)):
        self.num_games = num_games
        self.speed = speed
        self.grid_size = grid_size
        self.results = []
        self.current_game = 0
        
    def run_single_game(self) -> dict:
        """Run a single game and return performance stats."""
        print(f"Running game {self.current_game + 1}/{self.num_games}...", end="\r")
        
        # Create game controller with AI tracing enabled
        game_controller = GameController(
            speed=self.speed,
            ai=True,
            grid=self.grid_size,
            nes_mode=False,
            ai_tracing=False,  # Disable console logging for cleaner output
            auto_advance=True  # Enable full automation
        )
        
        # Run the game
        game_controller.run_game_loop()
        
        # Get AI performance stats
        ai_stats = game_controller.ai_controller.get_performance_stats()
        
        # Add game metadata
        game_stats = {
            'game_number': self.current_game + 1,
            'final_score': game_controller.game_state.score,
            'final_snake_length': len(game_controller.game_state.get_snake_body()),
            'game_duration_moves': ai_stats.get('total_moves', 0),
            'food_eaten': ai_stats.get('food_eaten', 0),
            'strategies_used': ai_stats.get('strategies_used', {}),
            'average_snake_length': ai_stats.get('average_snake_length', 0)
        }
        
        self.current_game += 1
        return game_stats
    
    def run_all_games(self):
        """Run all games and collect statistics."""
        print(f"Starting AI performance test: {self.num_games} games at speed {self.speed}")
        print("=" * 60)
        
        start_time = time.time()
        
        for i in range(self.num_games):
            try:
                game_stats = self.run_single_game()
                self.results.append(game_stats)
            except KeyboardInterrupt:
                print(f"\nTest interrupted after {i + 1} games.")
                break
            except Exception as e:
                print(f"\nError in game {i + 1}: {e}")
                continue
        
        total_time = time.time() - start_time
        print(f"\nCompleted {len(self.results)} games in {total_time:.1f} seconds")
        print("=" * 60)
    
    def analyze_results(self):
        """Analyze and display comprehensive results."""
        if not self.results:
            print("No results to analyze.")
            return
        
        print("=== AI PERFORMANCE ANALYSIS ===")
        print(f"Games completed: {len(self.results)}")
        
        # Extract key metrics
        scores = [r['final_score'] for r in self.results]
        snake_lengths = [r['final_snake_length'] for r in self.results]
        moves = [r['game_duration_moves'] for r in self.results]
        food_eaten = [r['food_eaten'] for r in self.results]
        
        # Calculate statistics
        print(f"\n📊 SCORE STATISTICS:")
        print(f"  Average score: {statistics.mean(scores):.1f}")
        print(f"  Median score: {statistics.median(scores):.1f}")
        print(f"  Best score: {max(scores)}")
        print(f"  Worst score: {min(scores)}")
        print(f"  Standard deviation: {statistics.stdev(scores):.1f}")
        
        print(f"\n🐍 SNAKE LENGTH STATISTICS:")
        print(f"  Average final length: {statistics.mean(snake_lengths):.1f}")
        print(f"  Median final length: {statistics.median(snake_lengths):.1f}")
        print(f"  Longest snake: {max(snake_lengths)}")
        print(f"  Shortest snake: {min(snake_lengths)}")
        
        print(f"\n⏱️ GAME DURATION STATISTICS:")
        print(f"  Average moves per game: {statistics.mean(moves):.1f}")
        print(f"  Median moves per game: {statistics.median(moves):.1f}")
        print(f"  Longest game: {max(moves)} moves")
        print(f"  Shortest game: {min(moves)} moves")
        
        print(f"\n🍎 FOOD COLLECTION STATISTICS:")
        print(f"  Average food eaten: {statistics.mean(food_eaten):.1f}")
        print(f"  Median food eaten: {statistics.median(food_eaten):.1f}")
        print(f"  Most food in one game: {max(food_eaten)}")
        print(f"  Least food in one game: {min(food_eaten)}")
        
        # Strategy analysis
        all_strategies = defaultdict(int)
        for result in self.results:
            for strategy, count in result['strategies_used'].items():
                all_strategies[strategy] += count
        
        print(f"\n🧠 STRATEGY USAGE:")
        total_moves = sum(all_strategies.values())
        for strategy, count in sorted(all_strategies.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_moves * 100) if total_moves > 0 else 0
            print(f"  {strategy}: {count} times ({percentage:.1f}%)")
        
        # Performance categories
        print(f"\n🏆 PERFORMANCE CATEGORIES:")
        high_scores = [s for s in scores if s >= 20]
        medium_scores = [s for s in scores if 10 <= s < 20]
        low_scores = [s for s in scores if s < 10]
        
        print(f"  High performers (20+ score): {len(high_scores)} games ({len(high_scores)/len(scores)*100:.1f}%)")
        print(f"  Medium performers (10-19 score): {len(medium_scores)} games ({len(medium_scores)/len(scores)*100:.1f}%)")
        print(f"  Low performers (<10 score): {len(low_scores)} games ({len(low_scores)/len(scores)*100:.1f}%)")
        
        print("=" * 60)
    
    def save_results(self, filename: str = "ai_performance_results.txt"):
        """Save detailed results to a file."""
        with open(filename, 'w') as f:
            f.write("AI Performance Test Results\n")
            f.write("=" * 50 + "\n\n")
            
            for result in self.results:
                f.write(f"Game {result['game_number']}:\n")
                f.write(f"  Score: {result['final_score']}\n")
                f.write(f"  Snake length: {result['final_snake_length']}\n")
                f.write(f"  Moves: {result['game_duration_moves']}\n")
                f.write(f"  Food eaten: {result['food_eaten']}\n")
                f.write(f"  Strategies: {result['strategies_used']}\n")
                f.write("\n")
        
        print(f"Detailed results saved to {filename}")


def main():
    """Main function to run the performance test."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Performance Tester")
    parser.add_argument("--games", type=int, default=50, help="Number of games to run (default: 50)")
    parser.add_argument("--speed", type=int, default=60, help="Game speed (default: 60)")
    parser.add_argument("--grid", type=int, nargs=2, default=[20, 20], help="Grid size (default: 20 20)")
    parser.add_argument("--save", action="store_true", help="Save detailed results to file")
    
    args = parser.parse_args()
    
    tester = AIPerformanceTester(
        num_games=args.games,
        speed=args.speed,
        grid_size=tuple(args.grid)
    )
    
    try:
        tester.run_all_games()
        tester.analyze_results()
        
        if args.save:
            tester.save_results()
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
        if tester.results:
            tester.analyze_results()


if __name__ == '__main__':
    main() 