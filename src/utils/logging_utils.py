import logging
import os
import sys
import json
from typing import Optional
from datetime import datetime

class JsonFormatter(logging.Formatter):
    def format(self, record):
        # Convert the log record to a dict
        log_record = {
            'timestamp': self.formatTime(record, self.datefmt),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        # If the message is already JSON, merge it
        try:
            msg_dict = json.loads(record.getMessage())
            log_record.update(msg_dict)
        except Exception:
            pass
        return json.dumps(log_record)

def setup_logging(log_to_file: bool = True, log_to_console: bool = True, log_level: str = 'INFO', log_dir: str = 'logs', log_name: str = 'game_session', json_mode: bool = False) -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    # Remove all handlers
    logger.handlers = []
    if json_mode:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    if log_to_file:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = os.path.join(log_dir, f'{log_name}_{timestamp}.log')
        file_handler = logging.FileHandler(log_path, mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger 