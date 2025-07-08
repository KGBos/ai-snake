import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import json
import logging
import tempfile
import shutil
import pytest
import glob
from src.utils.logging_utils import setup_logging
from ai.learning import LearningAIController
from game.models import GameState

@pytest.fixture
def temp_log_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)

@pytest.fixture(autouse=True)
def reset_logging():
    # Remove all handlers before each test
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    yield
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

def test_episode_end_json_logged(temp_log_dir):
    # Setup logging (will write to logs/ directory)
    setup_logging(log_to_file=True, log_to_console=False, log_level='INFO', json_mode=True)
    # Simulate episode end
    controller = LearningAIController(grid_size=(5, 5), training=True)
    controller.current_episode_reward = 42
    controller.current_episode_length = 10
    controller.food_eaten_this_episode = 3
    controller.deaths_this_episode = 1
    controller.agent.memory.append('dummy')
    controller.agent.epsilon = 0.5
    controller.record_episode_end(final_score=7, death_type='wall')
    # Find the most recent log file in logs/
    log_files = glob.glob('logs/game_session_*.log')
    assert log_files, 'No log files found in logs directory.'
    latest_log = max(log_files, key=os.path.getmtime)
    # Check log file for JSON entry
    with open(latest_log, 'r') as f:
        lines = f.readlines()
    json_lines = [l for l in lines if l.strip().startswith('{') and 'episode_end' in l]
    assert json_lines, 'No JSON episode_end log found.'
    entry = json.loads(json_lines[-1])
    assert entry['event'] == 'episode_end'
    assert entry['final_score'] == 7
    assert entry['death_type'] == 'wall'


def test_setup_logging_creates_file(temp_log_dir):
    log_file = os.path.join(temp_log_dir, 'test.log')
    setup_logging(log_to_file=True, log_to_console=False, log_level='INFO')
    logger = logging.getLogger()
    logger.info('Test log entry')
    # Find the log file in the logs directory
    logs_dir = 'logs'
    found = False
    if os.path.exists(logs_dir):
        for fname in os.listdir(logs_dir):
            if fname.endswith('.log'):
                found = True
                break
    assert found, 'No log file created in logs directory.'


def test_error_logging(temp_log_dir):
    setup_logging(log_to_file=False, log_to_console=True, log_level='INFO')
    logger = logging.getLogger('TestLogger')
    with pytest.raises(Exception):
        try:
            raise ValueError('Test error')
        except Exception as e:
            logger.error(f'Error occurred: {e}')
            raise 