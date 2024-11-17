import io
from chess.pgn import read_game
import chess
import matplotlib.pyplot as plt
import numpy as np
from model_data_processor import fetch_boards_until_limit, fetch_games, numeric_board
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from collections import defaultdict

from model_trainer import train_model


def train_model2(numeric_boards, moves):
    """
    Trains a deeper CNN model to predict the next move based on the numeric board states.

    Args:
        numeric_boards (array-like): Array of numeric board states (8x8).
        moves (array-like): Array of moves in UCI format corresponding to the board states.

    Returns:
        tf.keras.Model: The trained model.
        float: Accuracy of the model on the test set.
    """
    move_encoder = LabelEncoder()
    y_encoded = move_encoder.fit_transform(moves)

    X_numeric = np.array(numeric_boards)
    X_numeric = np.expand_dims(X_numeric, axis=-1)

    X_train, X_test, y_train, y_test = train_test_split(X_numeric, y_encoded, test_size=0.2, random_state=42)

    input_shape = X_numeric.shape[1:]
    output_dim = len(move_encoder.classes_)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(output_dim, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=15, batch_size=64, validation_data=(X_test, y_test))

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    model.move_encoder = move_encoder
    return model, accuracy


def main():
    username = "hikaru"
    time_control = "blitz"
    color = "black"
    month = 11  # Provide the required month argument for fetch_games

    # Fetch training data
    numeric_boards, moves = fetch_boards_until_limit(username, time_control, color, target_samples=10000)

    # Train two models
    model1, _ = train_model(numeric_boards, moves)
    model2, _ = train_model2(numeric_boards, moves)

    # Fetch games
    games_data = fetch_games(username, 2024, month)
    all_confidences_model1 = defaultdict(list)
    all_confidences_model2 = defaultdict(list)
    counter = 0
    for game_data in games_data["games"]:
        pgn = game_data["pgn"]
        print(f"Processing game: {counter}")
        counter += 1
        game = read_game(io.StringIO(pgn))
        board = game.board()

        is_model_turn = color.lower() == "white"

        for move in game.mainline_moves():
            if is_model_turn:
                input_board = np.expand_dims(np.expand_dims(numeric_board(board), axis=0), axis=-1)

                # Predictions for model1
                predictions_model1 = model1.predict(input_board)
                confidence_model1 = predictions_model1[0, np.argmax(predictions_model1)]
                all_confidences_model1[len(board.move_stack)].append(confidence_model1)

                # Predictions for model2
                predictions_model2 = model2.predict(input_board)
                confidence_model2 = predictions_model2[0, np.argmax(predictions_model2)]
                all_confidences_model2[len(board.move_stack)].append(confidence_model2)

            board.push(move)
            is_model_turn = not is_model_turn

    # Calculate average confidence at each depth for both models
    depths_model1 = sorted(all_confidences_model1.keys())
    avg_confidences_model1 = [np.mean(all_confidences_model1[depth]) for depth in depths_model1]

    depths_model2 = sorted(all_confidences_model2.keys())
    avg_confidences_model2 = [np.mean(all_confidences_model2[depth]) for depth in depths_model2]

    # Plot average confidence over depth for both models
    plt.figure(figsize=(12, 8))
    plt.plot(depths_model1, avg_confidences_model1, label="Model1 Average Confidence", marker="o")
    plt.plot(depths_model2, avg_confidences_model2, label="Model2 Average Confidence", marker="x")
    plt.xlabel("Depth")
    plt.ylabel("Confidence")
    plt.title("Average Model Confidence vs. Depth")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
