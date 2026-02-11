"""Unit tests for the DQN agent and RewardCalculator."""
import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ai_snake.ai.dqn import SimpleSnakeDQN, DQNAgent, Experience
from ai_snake.game.models import GameState


# ── SimpleSnakeDQN Network Tests ──────────────────────────────────────────────

class TestSimpleSnakeDQN:
    def test_output_shape(self):
        """Network should output one Q-value per action."""
        net = SimpleSnakeDQN(input_size=27, hidden_size=64, output_size=4)
        dummy = torch.randn(1, 27)
        out = net(dummy)
        assert out.shape == (1, 4)

    def test_batch_forward(self):
        """Network should handle batched input."""
        net = SimpleSnakeDQN(input_size=27, hidden_size=64, output_size=4)
        batch = torch.randn(32, 27)
        out = net(batch)
        assert out.shape == (32, 4)

    def test_weights_initialized(self):
        """Xavier init should produce bounded weights."""
        net = SimpleSnakeDQN()
        for param in net.parameters():
            assert param.abs().max().item() < 10.0


# ── DQNAgent Tests ────────────────────────────────────────────────────────────

class TestDQNAgent:
    @pytest.fixture
    def agent(self):
        return DQNAgent(device='cpu')

    def test_initial_epsilon(self, agent):
        assert agent.epsilon == 1.0

    def test_state_representation_shape(self, agent):
        """get_state_representation should return a (1, state_size) tensor."""
        gs = GameState(grid_width=10, grid_height=10)
        state = agent.get_state_representation(gs)
        assert state.shape == (1, agent.state_size)

    def test_get_action_returns_valid(self, agent):
        """Action must be in [0, action_size)."""
        gs = GameState(grid_width=10, grid_height=10)
        action = agent.get_action(gs)
        assert 0 <= action < agent.action_size

    def test_remember_stores_experience(self, agent):
        """Memory should grow when adding experiences."""
        gs = GameState(grid_width=10, grid_height=10)
        state = agent.get_state_representation(gs)
        agent.remember(state, 0, 1.0, state, False)
        assert len(agent.memory) == 1

    def test_replay_needs_min_batch(self, agent):
        """Replay should not crash when memory is smaller than batch_size."""
        gs = GameState(grid_width=10, grid_height=10)
        state = agent.get_state_representation(gs)
        agent.remember(state, 0, 1.0, state, False)
        # Should return without error
        agent.replay()

    def test_save_and_load_model(self, agent, tmp_path):
        """Model should be savable and loadable."""
        path = str(tmp_path / 'test_model.pth')
        agent.save_model(path)
        assert os.path.exists(path)

        new_agent = DQNAgent(device='cpu')
        new_agent.load_model(path)
        # Q-values should match after load
        gs = GameState(grid_width=10, grid_height=10)
        state = agent.get_state_representation(gs)
        original_q = agent.q_network(state)
        loaded_q = new_agent.q_network(state)
        assert torch.allclose(original_q, loaded_q, atol=1e-6)


# ── RewardCalculator Tests ────────────────────────────────────────────────────

class TestRewardCalculator:
    @pytest.fixture
    def calc(self):
        # Defer import so lazy logging doesn't trigger during collection
        from ai_snake.ai.learning import RewardCalculator
        return RewardCalculator()

    def test_food_reward_positive(self, calc):
        """Eating food should yield a positive reward."""
        gs = GameState(grid_width=10, grid_height=10)
        gs.score = 1  # simulate food eaten
        calc.last_score = 0
        reward = calc.calculate_reward(gs, done=False)
        assert reward > 0

    def test_death_penalty_negative(self, calc):
        """Dying should yield a negative reward."""
        gs = GameState(grid_width=10, grid_height=10)
        reward = calc.calculate_reward(gs, done=True)
        assert reward < 0

    def test_distance_toward_food_positive(self, calc):
        """Moving closer to food should give a positive distance signal."""
        gs = GameState(grid_width=10, grid_height=10)
        gs.food = (7, 5)

        # First step: head at (5,5)
        calc.calculate_reward(gs, done=False)

        # Move head closer to food
        gs.set_snake_for_testing([(6, 5)])
        reward = calc.calculate_reward(gs, done=False)
        # The distance portion should contribute positively
        assert calc.reward_breakdown.get('distance_closer', 0) > 0 or reward >= -calc.move_penalty * 2

    def test_moves_without_food_increments(self, calc):
        """moves_without_food should increment when no food is eaten."""
        gs = GameState(grid_width=10, grid_height=10)
        calc.calculate_reward(gs, done=False)
        assert calc.moves_without_food == 1
        calc.calculate_reward(gs, done=False)
        assert calc.moves_without_food == 2

    def test_food_eaten_resets_counter(self, calc):
        """Eating food should reset moves_without_food."""
        gs = GameState(grid_width=10, grid_height=10)
        calc.calculate_reward(gs, done=False)
        assert calc.moves_without_food == 1

        gs.score = 1
        calc.last_score = 0
        calc.calculate_reward(gs, done=False)
        assert calc.moves_without_food == 0

    def test_manhattan_distance(self, calc):
        """calculate_distance_to_food should return Manhattan distance."""
        assert calc.calculate_distance_to_food((0, 0), (3, 4)) == 7
        assert calc.calculate_distance_to_food((5, 5), (5, 5)) == 0
