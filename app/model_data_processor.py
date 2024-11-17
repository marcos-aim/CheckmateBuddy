import chess.pgn
import chess
import io
import numpy as np
from datetime import datetime

from chess_com import fetch_games


def numeric_board(board):
    """
    Converts a chess.Board object into a numeric 2D array representation.

    Args:
        board (chess.Board): The board state.

    Returns:
        np.ndarray: A numeric array of shape (8, 8), with pieces represented by integers.
    """
    piece_map = board.piece_map()
    board_array = np.zeros((8, 8), dtype=int)

    for square, piece in piece_map.items():
        row = 7 - (square // 8)  # Flip row for proper board orientation
        col = square % 8
        piece_value = piece.piece_type * (1 if piece.color == chess.WHITE else -1)
        board_array[row, col] = piece_value

    return board_array


def extract_numeric_boards_and_moves(pgn_string, color):
    """
    Extracts numeric board states and next moves from the PGN, respecting the game order logic for the given color.

    Args:
        pgn_string (str): The PGN string of the game.
        color (str): The color whose turn order is respected ("white" or "black").

    Returns:
        tuple: Two lists - numeric_boards (numeric board states) and moves (next moves).
    """
    pgn = chess.pgn.read_game(io.StringIO(pgn_string))
    if not pgn:
        raise ValueError("Invalid PGN string provided.")

    board = pgn.board()
    is_user_turn = color.lower() == "white"

    numeric_boards, moves = [], []

    for move in pgn.mainline_moves():
        if is_user_turn:
            numeric_boards.append(numeric_board(board))
            moves.append(move.uci())
        board.push(move)
        is_user_turn = not is_user_turn

    return numeric_boards, moves


def process_games_for_training(games, time_control, username, color):
    """
    Processes a list of games to extract training data based on time control and color.

    Args:
        games (list): List of game dictionaries with keys like 'pgn', 'time_class', and 'white'/'black'.
        time_control (str): The desired time control (e.g., 'blitz', 'rapid', 'bullet').
        username (str): The username to track (case-insensitive).
        color (str): The color to track ("white" or "black").

    Returns:
        tuple: Two lists - numeric_boards (numeric board states) and moves (next moves).
    """
    all_numeric_boards, all_moves = [], []

    for game in games:
        if game.get("time_class") != time_control:
            continue
        if color.lower() == "white" and game.get("white", {}).get("username", "").lower() != username.lower():
            continue
        if color.lower() == "black" and game.get("black", {}).get("username", "").lower() != username.lower():
            continue

        pgn = game.get("pgn")
        if not pgn:
            continue

        try:
            numeric_boards, moves = extract_numeric_boards_and_moves(pgn, color)
            all_numeric_boards.extend(numeric_boards)
            all_moves.extend(moves)
        except Exception as e:
            print(f"Error processing game {game.get('url', 'unknown')}: {e}")

    return all_numeric_boards, all_moves


def fetch_boards_until_limit(username, time_control, color, target_samples=10000, max_months=12):
    """
    Fetches numeric boards and moves, going backward in months until the target sample size is reached or the limit is exceeded.

    Args:
        username (str): The Chess.com username.
        time_control (str): Desired time control (e.g., 'blitz', 'rapid', 'bullet').
        color (str): The color to track ("white" or "black").
        target_samples (int): The minimum number of samples to collect.
        max_months (int): Maximum number of months to go back.

    Returns:
        tuple: Two lists - numeric_boards and moves.
    """
    current_year = datetime.now().year
    current_month = datetime.now().month
    all_numeric_boards, all_moves = [], []

    for _ in range(max_months):
        print(f"Fetching games for {current_month}/{current_year}...")
        try:
            data = fetch_games(username, str(current_year), f"{current_month:02d}")
            games = data.get("games", [])
            numeric_boards, moves = process_games_for_training(games, time_control, username, color)

            all_numeric_boards.extend(numeric_boards)
            all_moves.extend(moves)
            print(f"Collected {len(numeric_boards)} boards from {current_month}/{current_year}. Total so far: {len(all_numeric_boards)}")

            if len(all_numeric_boards) >= target_samples:
                break
        except Exception as e:
            print(f"Error fetching games for {current_month}/{current_year}: {e}")

        # Move to the previous month
        current_month -= 1
        if current_month == 0:  # Handle year decrement
            current_month = 12
            current_year -= 1

    if len(all_numeric_boards) < target_samples:
        print(f"Warning: Only {len(all_numeric_boards)} boards collected. Target was {target_samples}.")

    return all_numeric_boards, all_moves