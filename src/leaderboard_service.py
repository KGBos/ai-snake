from render.leaderboard import Leaderboard
from typing import Optional

class LeaderboardService:
    def __init__(self, file_path: Optional[str] = "leaderboard.json"):
        try:
            self.leaderboard = Leaderboard(file_path)
        except (IOError, OSError, ValueError, Exception) as e:
            print(f"Error loading leaderboard file: {e}")
            self.leaderboard = Leaderboard(file_path=None)
        self.session_leaderboard = Leaderboard(file_path=None)

    def add_entry(self, episode: int, reward: float, death_type: str, high_score: bool = False):
        self.leaderboard.add_entry(episode, reward, death_type, high_score=high_score)
        self.session_leaderboard.add_entry(episode, reward, death_type, high_score=high_score)

    def get_alltime_entries(self):
        return self.leaderboard.get_formatted_entries()

    def get_session_entries(self):
        return self.session_leaderboard.get_formatted_entries()

    def get_alltime_stats(self):
        return self.leaderboard.get_stats()

    def get_session_stats(self):
        return self.session_leaderboard.get_stats() 