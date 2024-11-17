import json
from os import environ as env
from urllib.parse import quote_plus, urlencode
import threading
import time
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import chess
import chess.engine
from flask import Flask, redirect, render_template, session, request, jsonify, url_for
from dotenv import find_dotenv, load_dotenv
from authlib.integrations.flask_client import OAuth
from model_data_processor import numeric_board

# Load the model and move encoder
loaded_model = load_model("app/chess_model.h5")
loaded_move_encoder = joblib.load("app/move_encoder.pkl")

app = Flask(__name__)

ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

app.secret_key = env.get("APP_SECRET_KEY")
REDIRECT_URL = env.get("REDIRECT_URL", "http://127.0.0.1:3000")

oauth = OAuth(app)

oauth.register(
    "auth0",
    client_id=env.get("AUTH0_CLIENT_ID"),
    client_secret=env.get("AUTH0_CLIENT_SECRET"),
    client_kwargs={"scope": "openid profile email"},
    server_metadata_url=f'https://{env.get("AUTH0_DOMAIN")}/.well-known/openid-configuration'
)

# Dictionary to manage boards for each user and their last access time
boards = {}

# Expiration settings
EXPIRATION_TIME = 3600  # 1 hour in seconds


def cleanup_boards():
    """Periodic cleanup of expired boards."""
    while True:
        current_time = time.time()
        keys_to_delete = []
        for user_id, data in boards.items():
            last_access_time = data['last_access']
            if current_time - last_access_time > EXPIRATION_TIME:
                keys_to_delete.append(user_id)

        for key in keys_to_delete:
            del boards[key]

        time.sleep(300)  # Run cleanup every 5 minutes


# Start the cleanup thread
cleanup_thread = threading.Thread(target=cleanup_boards, daemon=True)
cleanup_thread.start()


def create_stockfish_engine(elo):
    engine = chess.engine.SimpleEngine.popen_uci("app/stockfish/stockfish-windows-x86-64-vnni512.exe")
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
    return engine


stockfish_engine = create_stockfish_engine(1700)


@app.route("/")
def home():
    if "user_id" not in session:
        session["user_id"] = str(time.time()) + "-" + str(np.random.randint(1000))
    return render_template("index.html", session=session.get("user"), pretty=json.dumps(session.get("user"), indent=4))


@app.route("/play_move", methods=["POST"])
def play_specific_move():
    user_id = session.get("user_id")
    if user_id not in boards:
        boards[user_id] = {"board": chess.Board(), "last_access": time.time()}
    else:
        boards[user_id]["last_access"] = time.time()

    board = boards[user_id]["board"]
    data = request.json
    move_string = data.get("move")  # Example: "e2e4"
    try:
        move = chess.Move.from_uci(move_string)
        if move in board.legal_moves:
            board.push(move)
            if board.is_checkmate():
                return jsonify({"status": "checkmate", "board": board.fen()})
            return jsonify({"status": "success", "board": board.fen()})
        else:
            return jsonify({"status": "illegal"})
    except ValueError:
        return jsonify({"status": "invalid"})


@app.route("/move", methods=["POST"])
def make_move():
    user_id = session.get("user_id")
    if user_id not in boards:
        boards[user_id] = {"board": chess.Board(), "last_access": time.time()}
    else:
        boards[user_id]["last_access"] = time.time()

    board = boards[user_id]["board"]
    data = request.json
    source = data.get("from")
    target = data.get("to")

    try:
        move = chess.Move.from_uci(f"{source}{target}")
        if move in board.legal_moves:
            board.push(move)
            if board.is_checkmate():
                return jsonify({"status": "checkmate", "board": board.fen()})

            test_board = numeric_board(board)
            test_board_expanded = np.expand_dims(test_board, axis=(0, -1))
            predictions = loaded_model.predict(test_board_expanded)
            predicted_move_idx = np.argmax(predictions)
            predicted_move = loaded_move_encoder.inverse_transform([predicted_move_idx])

            stock_move = stockfish_engine.play(board, chess.engine.Limit(time=0.1)).move
            board.push(stock_move)

            return jsonify({"status": "success", "board": board.fen()})
        else:
            return jsonify({"status": "illegal"})
    except ValueError:
        return jsonify({"status": "invalid"})


@app.route("/reset", methods=["POST"])
def reset_board():
    user_id = session.get("user_id")
    boards[user_id] = {"board": chess.Board(), "last_access": time.time()}
    return jsonify({"status": "success"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
