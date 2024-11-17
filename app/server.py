import json
from os import environ as env
from urllib.parse import quote_plus, urlencode

from authlib.integrations.flask_client import OAuth
from dotenv import find_dotenv, load_dotenv

from flask import Flask, redirect, render_template, session, request, jsonify, url_for
import chess.engine
import time
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import chess

from model_data_processor import numeric_board

# load the model
global moves
loaded_model = load_model("app/chess_model.h5")
# Load the label encoder
loaded_move_encoder = joblib.load("app/move_encoder.pkl")

app = Flask(__name__)
board = chess.Board()  # Initialize the chess board

ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

app = Flask(__name__)
app.secret_key = env.get("APP_SECRET_KEY")
REDIRECT_URL = env.get("REDIRECT_URL", "http://127.0.0.1:3000")


oauth = OAuth(app)

oauth.register(
    "auth0",
    client_id=env.get("AUTH0_CLIENT_ID"),
    client_secret=env.get("AUTH0_CLIENT_SECRET"),
    client_kwargs={
        "scope": "openid profile email",
    },
    server_metadata_url=f'https://{env.get("AUTH0_DOMAIN")}/.well-known/openid-configuration'
)


@app.route("/login")
def login():
    # Append /callback to the base redirect URL
    redirect_url = f"{REDIRECT_URL}/callback"
    return oauth.auth0.authorize_redirect(
        redirect_uri=redirect_url
    )



@app.route("/callback", methods=["GET", "POST"])
def callback():
    token = oauth.auth0.authorize_access_token()
    session["user"] = token
    return redirect("/")


@app.route("/logout")
def logout():
    session.clear()
    print(url_for("home", _external=True))
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
    return render_template("index.html", session=session.get('user'), pretty=json.dumps(session.get('user'), indent=4))

def create_stockfish_engine(elo):
    engine = chess.engine.SimpleEngine.popen_uci("app/stockfish/stockfish-windows-x86-64-vnni512.exe")
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
    return engine
moves = 0
stockfish_engine = create_stockfish_engine(1700)

@app.route("/play_move", methods=["POST"])
def play_specific_move(moveString):
    global board
    try:
        # Define the specific move (e.g., "e2e4")
        move = chess.Move.from_uci(moveString)
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
    global board, moves
    #if not ready_to_play:
        #return jsonify("not ready to play")
    data = request.json
    source = data.get("from")  # Source square
    target = data.get("to")    # Target square
    fen = data.get("fen")

    try:
        print(fen)
        move = chess.Move.from_uci(f"{source}{target}")
        if move in board.legal_moves:
            board.push(move)

            if board.is_checkmate():
                return jsonify({"status": "checkmate", "board": board.fen()})

            test_board = numeric_board(board)  # Use the first numeric board as an example
            test_board_expanded = np.expand_dims(test_board, axis=(0, -1))  # Expand dims for batch and channels
            #print(test_board)
            #print(test_board_expanded);
            predictions = loaded_model.predict(test_board_expanded)
            predicted_move_idx = np.argmax(predictions)
            predicted_move = loaded_move_encoder.inverse_transform([predicted_move_idx])

            print(f"Predicted Move: {predicted_move[0]}")
            print(f"Confidence: {predictions[0, predicted_move_idx]:.2f}")
            roll = (1 - (moves/30) + np.random.rand());  

            stock_move = stockfish_engine.play(board, chess.engine.Limit(time=0.1)).move

            print(stock_move)
            moves += 1

        
            return jsonify({"status": "success", "board": board.fen()})

            """

            eng_move = chess.Move.from_uci(predicted_move)
            if eng_move in board.legal_moves:
                board.push(eng_move)

                if board.is_checkmate():
                    return jsonify({"status": "checkmate", "board": board.fen()})

                return jsonify({"status": "success", "board": board.fen()})
            else:
                return jsonify({"status": "illegal"})



"""
        else:
            return jsonify({"status": "illegal"})
    except ValueError:
        return jsonify({"status": "invalid"})
@app.route("/reset", methods=["POST"])
def reset_board():
    global board
    board = chess.Board()
    return jsonify({"status": "success"})

@app.route("/ready", methods=["POST"])
def ready_toggle():
    global ready_to_play
    ready_to_play = True
    return jsonify({"status": "success"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
