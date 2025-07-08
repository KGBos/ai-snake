import json
import os
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime
from game.models import GameState
from config.config import *
from config.loader import *

class LeaderboardEntry:
    """Represents a single leaderboard entry."""
    
    def __init__(self, episode: int, reward: float, death_type: str, timestamp: Optional[float] = None, high_score: bool = False):
        self.episode = episode
        self.reward = reward
        self.death_type = death_type
        self.timestamp = timestamp or datetime.now().timestamp()
        self.previous_rank: Optional[int] = None
        self.rank_change = 0
        self.high_score = high_score
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'episode': self.episode,
            'reward': self.reward,
            'death_type': self.death_type,
            'timestamp': self.timestamp,
            'previous_rank': self.previous_rank,
            'rank_change': self.rank_change,
            'high_score': self.high_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'LeaderboardEntry':
        """Create from dictionary."""
        entry = cls(data['episode'], data['reward'], data['death_type'], data.get('timestamp'), data.get('high_score', False))
        entry.previous_rank = data.get('previous_rank')
        entry.rank_change = data.get('rank_change', 0)
        return entry

class Leaderboard:
    """Manages the top 10 episodes leaderboard with Billboard 100-style formatting."""
    
    def __init__(self, file_path: Optional[str] = "leaderboard.json"):
        self.file_path = file_path
        self.entries: List[LeaderboardEntry] = []
        if self.file_path is not None:
            self.load_leaderboard()
        else:
            self.entries = []
    
    def load_leaderboard(self):
        """Load leaderboard from file."""
        if self.file_path is None:
            return
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r') as f:
                    data = json.load(f)
                    self.entries = [LeaderboardEntry.from_dict(entry) for entry in data]
            except (json.JSONDecodeError, KeyError):
                self.entries = []
        else:
            self.entries = []
    
    def save_leaderboard(self):
        """Save leaderboard to file."""
        if self.file_path is None:
            return
        try:
            with open(self.file_path, 'w') as f:
                json.dump([entry.to_dict() for entry in self.entries], f, indent=2)
        except IOError:
            pass  # Silently fail if we can't save
    
    def add_entry(self, episode: int, reward: float, death_type: str, high_score: bool = False) -> bool:
        """Add a new entry and return True if it made the leaderboard."""
        # Store previous ranks before adding new entry
        for i, entry in enumerate(self.entries):
            entry.previous_rank = i + 1
        
        # Remove any previous entry for this episode (one line per episode)
        self.entries = [e for e in self.entries if e.episode != episode]
        
        # Create new entry
        new_entry = LeaderboardEntry(episode, reward, death_type, high_score=high_score)
        
        # Add to list and sort by reward (descending)
        self.entries.append(new_entry)
        self.entries.sort(key=lambda x: x.reward, reverse=True)
        
        # Keep only top 10
        if len(self.entries) > 10:
            self.entries = self.entries[:10]
        
        # Find the new entry and calculate rank change
        for i, entry in enumerate(self.entries):
            if entry.episode == episode and entry.reward == reward:
                entry.rank_change = (entry.previous_rank - (i + 1)) if entry.previous_rank else 0
                break
        
        # Save to file
        self.save_leaderboard()
        
        # Return True if this entry is in the top 10
        return new_entry in self.entries
    
    def get_formatted_entries(self) -> List[Dict]:
        """Get formatted entries for display."""
        formatted = []
        for i, entry in enumerate(self.entries):
            # Determine arrow and color
            if entry.rank_change > 0:
                arrow = "↑"
                color = (100, 255, 100)  # Green
                change_text = f"+{entry.rank_change}"
            elif entry.rank_change < 0:
                arrow = "↓"
                color = (255, 100, 100)  # Red
                change_text = f"{entry.rank_change}"
            else:
                arrow = "→"
                color = (255, 255, 255)  # White
                change_text = "NEW" if entry.previous_rank is None else "0"
            
            # Format death type
            death_abbrev = {
                'wall': 'WALL',
                'self': 'SELF', 
                'starvation': 'STARVE',
                'other': 'OTHER'
            }.get(entry.death_type.lower(), entry.death_type.upper())
            
            formatted.append({
                'rank': i + 1,
                'episode': entry.episode,
                'reward': entry.reward,
                'death_type': death_abbrev,
                'arrow': arrow,
                'color': color,
                'change_text': change_text,
                'rank_change': entry.rank_change,
                'high_score': getattr(entry, 'high_score', False)
            })
        
        return formatted
    
    def get_stats(self) -> Dict:
        """Get leaderboard statistics."""
        if not self.entries:
            return {}
        
        rewards = [entry.reward for entry in self.entries]
        return {
            'total_entries': len(self.entries),
            'highest_reward': max(rewards),
            'average_reward': sum(rewards) / len(rewards),
            'lowest_reward': min(rewards)
        } 