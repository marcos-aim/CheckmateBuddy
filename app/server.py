import json
import time
import threading
from os import environ as env
from urllib.parse import quote_plus, urlencode

from authlib.integrations.flask_client import OAuth
from dotenv import find_dotenv, load_dotenv
from flask import Flask, redirect, render_template, session, request, jsonify, url_for
import chess.engine
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import chess

from model_data_processor import numeric_board

# Load the model
loaded_model = load_model("app/chess_model.h5")
loaded_move_encoder = joblib.load("app/move_encoder.pkl")

app = Flask(__name__)

# Load environment variables
ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

app.secret_key = env.get("APP_SECRET_KEY")
REDIRECT_URL = env.get("REDIRECT_URL", "http://127.0.0.1:3000")

# OAuth setup
oauth = OAuth(app)
oauth.register(
    "auth0",
    client_id=env.get("AUTH0_CLIENT_ID"),
    client_secret=env.get("AUTH0_CLIENT_SECRET"),
    client_kwargs={"scope": "openid profile email"},
    server_metadata_url=f'https://{env.get("AUTH0_DOMAIN")}/.well-known/openid-configuration'
)

# Board and session management
boards = {}
last_active = {}
SESSION_TIMEOUT = 3600  # Timeout in seconds (e.g., 1 hour)

def cleanup_inactive_sessions():
    """Periodically remove inactive sessions."""
    while True:
        current_time = time.time()
        to_remove = [
            user_id for user_id, last_time in last_active.items()
            if current_time - last_time > SESSION_TIMEOUT
        ]
        for user_id in to_remove:
            boards.pop(user_id, None)
            last_active.pop(user_id, None)
        time.sleep(300)  # Run cleanup every 5 minutes

# Start cleanup thread
cleanup_thread = threading.Thread(target=cleanup_inactive_sessions, daemon=True)
cleanup_thread.start()

def create_stockfish_engine(elo):
    """Initialize Stockfish engine with a given ELO."""
    engine = chess.engine.SimpleEngine.popen_uci("app/stockfish/stockfish-windows-x86-64-vnni512.exe")
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
    return engine

stockfish_engine = create_stockfish_engine(1700)

@app.route("/login")
def login():
    """Redirect to the Auth0 login page."""
    redirect_url = f"{REDIRECT_URL}/callback"
    return oauth.auth0.authorize_redirect(redirect_uri=redirect_url)

@app.route("/callback", methods=["GET", "POST"])
def callback():
    """Handle the callback from Auth0 after login."""
    token = oauth.auth0.authorize_access_token()
    session["user"] = token
    session["user_id"] = token.get("userinfo", {}).get("sub", str(time.time()))  # Use unique ID for logged-in users
    return redirect("/")

@app.route("/logout")
def logout():
    """Log out the user and clear the session."""
    session.clear()
    return redirect(
        "https://" + env.get("AUTH0_DOMAIN")
        + "/v2/logout?"
        + urlencode(
            {
                "returnTo": REDIRECT_URL,
                "client_id": env.get("AUTH0_CLIENT_ID"),
            },
            quote_via=quote_plus,
        )
    )

@app.route("/")
def home():
    """Home route."""
    if "user_id" not in session:
        session["user_id"] = str(time.time()) + "-" + str(np.random.randint(1000))  # Unique ID for non-logged-in users
    user_id = session["user_id"]
    last_active[user_id] = time.time()  # Update last active time
    return render_template("index.html", session=session.get("user"), pretty=json.dumps(session.get("user"), indent=4))

@app.route("/play_move", methods=["POST"])
def play_specific_move():
    """Handle a specific move by the user."""
    user_id = session.get("user_id")
    last_active[user_id] = time.time()  # Update last active time

    if user_id not in boards:
        boards[user_id] = chess.Board()  # Initialize board for the user
    board = boards[user_id]

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
    """Handle a user move and make a prediction or Stockfish move."""
    user_id = session.get("user_id")
    last_active[user_id] = time.time()  # Update last active time

    if user_id not in boards:
        boards[user_id] = chess.Board()  # Initialize board for the user
    board = boards[user_id]

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
    """Reset the user's chess board."""
    user_id = session.get("user_id")
    last_active[user_id] = time.time()  # Update last active time
    boards[user_id] = chess.Board()  # Reset user's board
    return jsonify({"status": "success"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
