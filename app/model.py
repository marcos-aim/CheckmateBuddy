import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf

def train_model(numeric_boards, moves):
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