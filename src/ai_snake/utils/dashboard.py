from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich.console import Console
from datetime import datetime

class TrainingDashboard:
    def __init__(self, total_episodes=5000):
        self.total_episodes = total_episodes
        self.layout = Layout()
        self.console = Console()
        self.live = None
        
        # State
        self.episode_count = 0
        self.best_score = 0
        self.recent_scores = []
        self.recent_logs = []
        
        # Initialize Layout
        self.layout.split(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=10)
        )
        self.layout["main"].split_row(
            Layout(name="stats", ratio=1),
            Layout(name="trend", ratio=2)
        )
        
    def start(self):
        """Start the live dashboard."""
        self.live = Live(self.layout, refresh_per_second=4, console=self.console)
        self.live.start()
        
    def stop(self):
        """Stop the live dashboard."""
        if self.live:
            self.live.stop()
            
    def update(self, stats):
        """Update dashboard with new stats."""
        self.episode_count = stats.get('episode', 0)
        current_score = stats.get('current_score', 0)
        epsilon = stats.get('epsilon', 0)
        memory = stats.get('memory', 0)
        avg_score = stats.get('avg_score', 0)
        
        if current_score > self.best_score:
            self.best_score = current_score
            
        self.recent_scores.append(current_score)
        if len(self.recent_scores) > 50:
            self.recent_scores.pop(0)

        # Update Log
        if stats.get('game_over', False):
             death_type = stats.get('death_type', 'unknown')
             timestamp = datetime.now().strftime("%H:%M:%S")
             log_entry = f"[{timestamp}] Ep {self.episode_count}: Score={current_score} ({death_type})"
             self.recent_logs.append(log_entry)
             if len(self.recent_logs) > 8:
                 self.recent_logs.pop(0)

        # 1. Header
        self.layout["header"].update(
             Panel(Text(f"🐍 AI SNAKE TRAINING - Headless Mode ({self.episode_count}/{self.total_episodes})", justify="center", style="bold green"))
        )
        
        # 2. Stats Table
        stats_table = Table(show_header=False, expand=True)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="bold white")
        
        stats_table.add_row("Episode", f"{self.episode_count}")
        stats_table.add_row("Exploration", f"{epsilon:.3f}")
        stats_table.add_row("Memory", f"{memory}")
        stats_table.add_row("Best Score", f"{self.best_score}", style="green")
        stats_table.add_row("Avg Score (50)", f"{avg_score:.1f}", style="yellow")
        
        self.layout["stats"].update(Panel(stats_table, title="Statistics", border_style="blue"))
        
        # 3. Trend (Sparkline-ish visualization using text)
        trend_text = self._generate_sparkline()
        self.layout["trend"].update(Panel(trend_text, title="Score Trend (Last 50 Games)", border_style="magenta"))
        
        # 4. Footer (Logs)
        log_text = "\n".join(self.recent_logs)
        self.layout["footer"].update(Panel(log_text, title="Recent Activity", border_style="white"))

    def _generate_sparkline(self):
        if not self.recent_scores:
            return Text("Waiting for data...", style="dim")
        
        # Normalize scores to height 10
        max_s = max(self.recent_scores) if self.recent_scores else 1
        if max_s == 0: max_s = 1
        
        # Simple text representation
        rows = []
        graph_height = 8
        
        # Create a grid
        grid = [[' ' for _ in range(50)] for _ in range(graph_height)]
        
        for x, score in enumerate(self.recent_scores):
            # Calculate height
            h = int((score / max_s) * (graph_height - 1))
            # Fill form bottom
            for y in range(h + 1):
                char = '█' if y == h else '│'
                grid[graph_height - 1 - y][x] = char
                
        # Convert to string
        final_str = ""
        for row in grid:
            final_str += "".join(row) + "\n"
            
        return Text(final_str, style="green")
