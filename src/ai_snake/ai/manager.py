# FIXME: Review this file for potential issues or improvements
from ai_snake.ai.rule_based import AIController
from ai_snake.ai.learning import LearningAIController, RewardCalculator
from ai_snake.config.loader import load_config, CONFIG_FILE

class AIManager:
    def __init__(self, grid_size, ai_tracing=False, learning_ai=False, model_path=None, config_path=CONFIG_FILE, starvation_threshold=50, log_to_file=False, wandb_logger=None):
        self.ai_controller = AIController(enable_tracing=ai_tracing, log_to_file=log_to_file)
        self.learning_ai_controller = None
        self.learning_ai = learning_ai
        self.model_path = model_path
        self.grid_size = grid_size
        self.wandb_logger = wandb_logger
        self.reward_calculator = None
        if learning_ai:
            self.learning_ai_controller = LearningAIController(grid_size=grid_size, model_path=model_path, training=True, wandb_logger=wandb_logger)
            self.reward_calculator = RewardCalculator(load_config(config_path), starvation_threshold=starvation_threshold)
        else:
            self.reward_calculator = None

    def make_move(self, game_state, use_learning_ai=False, manual_teaching_mode=False):
        if use_learning_ai and self.learning_ai_controller:
            if manual_teaching_mode:
                # Manual input overrides AI in teaching mode
                return None
            direction = self.learning_ai_controller.get_action(game_state)
            game_state.set_direction(direction, force=True)
        else:
            self.ai_controller.make_move(game_state)

    def get_action(self, game_state):
        if self.learning_ai_controller:
            return self.learning_ai_controller.get_action(game_state)
        return None

    def record_step(self, game_state, reward, done):
        if self.learning_ai_controller:
            self.learning_ai_controller.record_step(game_state, reward, done)

    def check_food_eaten(self, game_state):
        self.ai_controller.check_food_eaten(game_state)

    def get_stats(self):
        if self.learning_ai_controller:
            return self.learning_ai_controller.get_stats()
        return {}

    def save_model(self, filepath):
        if self.learning_ai_controller:
            self.learning_ai_controller.save_model(filepath)

    def load_model(self, filepath):
        if self.learning_ai_controller:
            self.learning_ai_controller.load_model(filepath)

    def record_episode_end(self, final_score, death_type=None):
        if self.learning_ai_controller:
            self.learning_ai_controller.record_episode_end(final_score, death_type)

    def toggle_learning_ai(self):
        self.learning_ai = not self.learning_ai
        if self.learning_ai and not self.learning_ai_controller:
            self.learning_ai_controller = LearningAIController(grid_size=self.grid_size, model_path=self.model_path, training=True, wandb_logger=self.wandb_logger)
            config = load_config(CONFIG_FILE)
            self.reward_calculator = RewardCalculator(config)

    def set_training_mode(self, training: bool):
        if self.learning_ai_controller:
            self.learning_ai_controller.set_training_mode(training) 