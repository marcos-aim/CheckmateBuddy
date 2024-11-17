import random

from model import train_model
from model_data_processor import fetch_boards_until_limit


def validate_time_control(time_control):
    """
    Validate the time control.
    """
    valid_time_controls = ["bullet", "blitz", "rapid"]
    if time_control not in valid_time_controls:
        raise ValueError(f"Invalid time control: {time_control}. Must be one of {valid_time_controls}.")
    return time_control


def validate_color(color):
    """
    Validate and determine the color to play.
    """
    valid_colors = ["white", "black", "random"]
    if color not in valid_colors:
        raise ValueError(f"Invalid color: {color}. Must be one of {valid_colors}.")
    if color == "random":
        return random.choice(["white", "black"])
    return color


def train_chess_model(username, time_control, color, target_samples=10000, max_months=12):
    """
    Validate inputs, fetch boards and moves, and train the model.

    Parameters:
    - username (str): The username of the chess player.
    - time_control (str): The time control for the games (bullet, blitz, rapid).
    - color (str): The color to play as (white, black, random).
    - target_samples (int): The target number of samples to fetch.
    - max_months (int): The maximum number of months to fetch data for.
    """
    # Validate inputs
    time_control = validate_time_control(time_control)
    color = validate_color(color)

    print(f"Fetching data for user '{username}', time control '{time_control}', color '{color}'...")

    # Fetch boards and moves
    all_numeric_boards, all_moves = fetch_boards_until_limit(username, time_control, color, target_samples, max_months)

    if not all_numeric_boards or not all_moves:
        raise ValueError("No data fetched. Ensure the parameters are correct and data exists.")

    print(f"Fetched {len(all_numeric_boards)} boards and {len(all_moves)} moves. Training model...")

    model, accuracy = train_model(all_numeric_boards, all_moves)

    print("Model training completed.")

    return model, accuracy