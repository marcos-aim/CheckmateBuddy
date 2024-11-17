from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from app.model_data_processor import fetch_boards_until_limit


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


