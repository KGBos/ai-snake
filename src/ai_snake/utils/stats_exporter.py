import json
import os
from datetime import datetime
import time

class StatsExporter:
    def __init__(self, output_file="dashboard_data.json"):
        self.output_file = output_file
        self.start_time = time.time()
        self.data = {
            "overview": {
                "episode": 0,
                "epsilon": 1.0,
                "total_steps": 0,
                "session_duration": 0
            },
            "performance": {
                "current_score": 0,
                "best_score": 0,
                "avg_score_50": 0
            },
            "trends": [],  # Last 100 scores
            "distribution": {},
            "recent_games": [], # Last 10 games
            "achievements": []
        }
        
        # Achievement definitions
        self.achievements_def = {
            "first_blood": {"name": "First Blood", "desc": "Complete 10 Episodes", "icon": "🩸", "unlocked": False},
            "glutton": {"name": "Glutton", "desc": "Eat 500 Food Total", "icon": "🍎", "unlocked": False},
            "century": {"name": "Century Club", "desc": "Score > 100 in one game", "icon": "💯", "unlocked": False},
            "big_brain": {"name": "Big Brain", "desc": "Reach Epsilon < 0.1", "icon": "🧠", "unlocked": False},
            "speed_demon": {"name": "Speed Demon", "desc": "1000 Episodes in session", "icon": "⚡", "unlocked": False},
            "survivor": {"name": "Survivor", "desc": "Survive 500 steps in one game", "icon": "🛡️", "unlocked": False}
        }
        
    def update(self, stats):
        """Update stats and write to file."""
        # 1. Update basic stats
        self.data["overview"]["episode"] = stats.get('episode', 0)
        self.data["overview"]["epsilon"] = stats.get('epsilon', 0)
        self.data["overview"]["total_steps"] += stats.get('steps_this_game', 0)
        self.data["overview"]["session_duration"] = int(time.time() - self.start_time)
        
        current_score = stats.get('current_score', 0)
        if current_score > self.data["performance"]["best_score"]:
            self.data["performance"]["best_score"] = current_score
        
        self.data["performance"]["current_score"] = current_score
        self.data["performance"]["avg_score_50"] = stats.get('avg_score', 0)
        
        # 2. Update Trends
        self.data["trends"].append({
            "episode": stats.get('episode', 0),
            "score": current_score,
            "epsilon": stats.get('epsilon', 0)
        })
        if len(self.data["trends"]) > 100:
            self.data["trends"].pop(0)
            
        # 3. Update Recent Games
        death_type = stats.get('death_type', 'unknown')
        self.data["recent_games"].insert(0, {
            "epid": stats.get('episode', 0),
            "score": current_score,
            "death": death_type,
            "time": datetime.now().strftime("%H:%M:%S")
        })
        if len(self.data["recent_games"]) > 10:
             self.data["recent_games"].pop()
             
        # 4. Update Distribution
        if death_type not in self.data["distribution"]:
            self.data["distribution"][death_type] = 0
        self.data["distribution"][death_type] += 1
        
        # 5. Check Achievements
        self._check_achievements(stats)
        
        # Write to file (atomic write to prevent read errors)
        self._write_file()
        
    def _check_achievements(self, stats):
        # 1. First Blood
        if not self.achievements_def["first_blood"]["unlocked"] and stats.get('episode', 0) >= 10:
             self._unlock("first_blood")
             
        # 2. Glutton
        # This needs total food eaten, which we don't track perfectly here but can approximate
        # ... simplifying for now
        
        # 3. Century
        if not self.achievements_def["century"]["unlocked"] and stats.get('current_score', 0) >= 100:
             self._unlock("century")
             
        # 4. Big Brain
        if not self.achievements_def["big_brain"]["unlocked"] and stats.get('epsilon', 1.0) < 0.1:
             self._unlock("big_brain")
             
        # 5. Speed Demon
        if not self.achievements_def["speed_demon"]["unlocked"] and stats.get('episode', 0) >= 1000:
             self._unlock("speed_demon")

        # Update data list
        self.data["achievements"] = [
            v for k, v in self.achievements_def.items() 
        ]
             
    def _unlock(self, key):
        self.achievements_def[key]["unlocked"] = True
        self.achievements_def[key]["unlock_time"] = datetime.now().strftime("%H:%M:%S")
        
    def _write_file(self):
        temp_file = self.output_file + ".tmp"
        try:
            with open(temp_file, 'w') as f:
                json.dump(self.data, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_file, self.output_file)
        except Exception as e:
            print(f"Error writing stats: {e}")
