import wandb
import os
import numpy as np
import logging

class WandbLogger:
    def __init__(self, project_name="ai-snake", config=None, debug=False):
        self.enabled = not debug
        self.logger = logging.getLogger(__name__)
        
        if self.enabled:
            try:
                # Initialize WandB
                wandb.init(project=project_name, config=config, reinit=True)
                self.logger.info("WandB initialized successfully.")
            except Exception as e:
                self.logger.error(f"Failed to initialize WandB: {e}")
                self.enabled = False
    
    def log(self, metrics, step=None):
        """Log key-value metrics."""
        if not self.enabled:
            return
        
        try:
            if step is not None:
                wandb.log(metrics, step=step)
            else:
                wandb.log(metrics)
        except Exception as e:
            self.logger.error(f"Failed to log to WandB: {e}")

    def log_histogram(self, name, data, step=None):
        """Log a histogram of data."""
        if not self.enabled:
            return
            
        try:
            wandb.log({name: wandb.Histogram(data)}, step=step)
        except Exception as e:
            self.logger.error(f"Failed to log histogram: {e}")

    def log_video(self, name, frames, fps=10, step=None):
        """Log a video from a list of numpy frames (H, W, C)."""
        if not self.enabled or not frames:
            return
            
        try:
            # WandB expects (T, C, H, W) for video
            # Input frames are usually (H, W, C) from Pygame/Numpy
            video_array = np.array(frames)
            
            # Transpose to (T, C, H, W)
            video_array = np.transpose(video_array, (0, 3, 1, 2))
            
            wandb.log({name: wandb.Video(video_array, fps=fps, format="mp4")}, step=step)
        except Exception as e:
            self.logger.error(f"Failed to log video: {e}")

    def finish(self):
        """Finish the run."""
        if self.enabled:
            wandb.finish()
