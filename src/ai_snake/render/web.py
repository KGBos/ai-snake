# FIXME: Review this file for potential issues or improvements
from ai_snake.render.base import BaseRenderer
from ai_snake.game.models import GameState
from typing import Optional
from flask import Flask, jsonify
import threading

class WebRenderer(BaseRenderer):
    def __init__(self, port: int = 5000):
        self.latest_state = None
        self.latest_info = None
        self.latest_time = None
        self.app = Flask(__name__)
        self.port = port
        self._setup_routes()
        self._start_server()

    def _setup_routes(self):
        @self.app.route('/state')
        def get_state():
            if self.latest_state is None:
                return jsonify({'status': 'waiting'})
            return jsonify({
                'snake': list(self.latest_state.snake),
                'food': self.latest_state.food,
                'score': self.latest_state.score,
                'info': self.latest_info,
                'time': self.latest_time
            })

    def _start_server(self):
        thread = threading.Thread(target=self.app.run, kwargs={'port': self.port, 'use_reloader': False})
        thread.daemon = True
        thread.start()

    def render(self, game_state: GameState, current_time: int, info: Optional[dict] = None):
        self.latest_state = game_state
        self.latest_info = info
        self.latest_time = current_time

    def clear_screen(self):
        pass  # No-op for web 