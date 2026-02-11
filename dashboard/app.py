from flask import Flask, render_template, jsonify
import json
import os
import subprocess

app = Flask(__name__)
DATA_FILE = os.path.join(os.path.dirname(__file__), '..', 'dashboard_data.json')

@app.route('/')
def index():
    return render_template('index.html')

import time

@app.route('/api/stats')
def get_stats():
    # Retry logic to handle potential race conditions during atomic write
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if os.path.exists(DATA_FILE):
                with open(DATA_FILE, 'r') as f:
                    data = json.load(f)
                response = jsonify(data)
                response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
                response.headers["Pragma"] = "no-cache"
                response.headers["Expires"] = "0"
                return response
        except (json.JSONDecodeError, OSError):
            # If file is empty or locked, wait briefly and retry
            time.sleep(0.05)
            continue
        except Exception as e:
            return jsonify({"error": str(e)}), 500
            
    return jsonify({"error": "Failed to read data after retries"}), 500
    return jsonify({"error": "No data found"}), 404

# Training Process Management
training_process = None

@app.route('/api/training/status', methods=['GET'])
def training_status():
    global training_process
    is_running = training_process is not None and training_process.poll() is None
    return jsonify({"running": is_running})

@app.route('/api/training/start', methods=['POST'])
def start_training():
    global training_process
    if training_process is not None and training_process.poll() is None:
        return jsonify({"message": "Training already running"}), 400
    
    try:
        # Run the training script as a subprocess
        # We use the same command as train_fast.sh but directly
        cmd = ["python3", "scripts/main.py", "--config", "config/fast_learning.yaml", "play", "--learning", "--headless", "--auto-advance", "--log"]
        training_process = subprocess.Popen(cmd,  cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return jsonify({"message": "Training started", "pid": training_process.pid})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/training/stop', methods=['POST'])
def stop_training():
    global training_process
    if training_process is None or training_process.poll() is not None:
        return jsonify({"message": "Training not running"}), 400
    
    try:
        training_process.terminate()
        training_process = None
        return jsonify({"message": "Training stopped"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
