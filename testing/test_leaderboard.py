import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from render.leaderboard import Leaderboard, LeaderboardEntry
from src.utils.logging_utils import setup_logging

setup_logging(log_to_file=False, log_to_console=True, log_level='INFO')

def test_add_and_sort_entries():
    lb = Leaderboard(file_path=None)
    lb.add_entry(1, 50, 'wall')
    lb.add_entry(2, 100, 'self')
    lb.add_entry(3, 75, 'starvation')
    entries = lb.get_formatted_entries()
    rewards = [e['reward'] for e in entries]
    assert rewards == sorted(rewards, reverse=True)
    assert entries[0]['reward'] == 100
    assert entries[1]['reward'] == 75
    assert entries[2]['reward'] == 50

def test_leaderboard_stats():
    lb = Leaderboard(file_path=None)
    lb.add_entry(1, 10, 'wall')
    lb.add_entry(2, 20, 'self')
    stats = lb.get_stats()
    assert stats['total_entries'] == 2
    assert stats['highest_reward'] == 20
    assert stats['lowest_reward'] == 10
    assert stats['average_reward'] == 15

def test_entry_to_from_dict():
    entry = LeaderboardEntry(5, 42.5, 'self', high_score=True)
    d = entry.to_dict()
    entry2 = LeaderboardEntry.from_dict(d)
    assert entry.episode == entry2.episode
    assert entry.reward == entry2.reward
    assert entry.death_type == entry2.death_type
    assert entry.high_score == entry2.high_score 