from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from model_data_processor import fetch_boards_until_limit


def train_model(numeric_boards, moves):
    """
    Trains a CNN model to predict the next move based on the numeric board states.

    Args:
        numeric_boards (array-like): Array of numeric board states (8x8).
        moves (array-like): Array of moves in UCI format corresponding to the board states.

    Returns:
        tf.keras.Model: The trained model.
    """
    # Encode moves (Y) as integers
    move_encoder = LabelEncoder()
    y_encoded = move_encoder.fit_transform(moves)

    # Convert numeric boards (X) into numpy array
    X_numeric = np.array(numeric_boards)

    # Expand dimensions to match CNN input requirements (8x8x1)
    X_numeric = np.expand_dims(X_numeric, axis=-1)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_numeric, y_encoded, test_size=0.2, random_state=42)

    # Define the CNN model
    input_shape = X_numeric.shape[1:]
    output_dim = len(move_encoder.classes_)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),  # Explicit input layer
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(output_dim, activation='softmax')  # Output probabilities for moves
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Save the label encoder for move decoding
    model.move_encoder = move_encoder

    return model, accuracy


def main():
    username = "marcosaim"
    time_control = "blitz"
    color = "black"

    # Variables to store results
    sample_sizes = list(range(1000, 15001, 1000))  # Start from 1,000 and go up to 60,000 in steps of 5,000
    accuracies = []
    confidences = []

    for target_samples in sample_sizes:
        # Fetch training data
        numeric_boards, moves = fetch_boards_until_limit(username, time_control, color, target_samples=target_samples)
        print(f"Number of training samples: {len(numeric_boards)}")

        # Train the model
        model, accuracy = train_model(numeric_boards, moves)
        accuracies.append(accuracy)

        # Test confidence on the first test board
        test_board = numeric_boards[0]  # Use the first numeric board as an example
        test_board_expanded = np.expand_dims(test_board, axis=(0, -1))  # Expand dims for batch and channels
        predictions = model.predict(test_board_expanded)
        predicted_move_idx = np.argmax(predictions)
        confidence = predictions[0, predicted_move_idx]
        confidences.append(confidence)

        print(f"Target Samples: {target_samples}, Test Accuracy: {accuracy * 100:.2f}%, Confidence: {confidence:.2f}")

    # Plot accuracy and confidence
    plt.figure(figsize=(10, 6))
    plt.plot(sample_sizes, accuracies, label="Accuracy", marker='o')
    plt.plot(sample_sizes, confidences, label="Confidence", marker='x')
    plt.xlabel("Number of Training Samples")
    plt.ylabel("Metric Value")
    plt.title("Accuracy and Confidence vs. Training Samples")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()