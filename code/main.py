"""
File: main.py
==========================================================================
Description:
This file contains the implementation of a Convolutional Neural Network (CNN)
using the Keras framework. The dataset used is a collection of audio files
representing major and minor chords. The goal is to classify the chords as
either "major" or "minor" based on their spectrograms.

==========================================================================
Date                    Author                   Description
10/22/2024         J. Riosmena          First implementation
11/18/2024          J. Riosmena         Updated implementation for better
                                                              performance

==========================================================================
Comments:

==========================================================================
To run:
$ python main.py

"""

# Libraries needed
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model

# Check if a GPU is available
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print(f"GPU: {physical_devices[0]}")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("No GPU found")

# Define the path to save the best model
best_model_path = 'models/best_model.keras'

def compare_models(new_model, old_model, x_val, y_val):
    """
    This function compares a new model with an old model based on their
    accuracy on a validation set. If the new model has a higher accuracy,
    it saves the new model to the specified path. Otherwise, it keeps the
    old model.

    Parameters:
    - new_model (tf.keras.Model): new model to compare
    - old_model (str): path to the old model
    - x_val (np.ndarray): validation set features
    - y_val (np.ndarray): validation set labels
    """

    # Check if the old model exists and compare the accuracies
    if os.path.exists(old_model):
        best_model = load_model(old_model)
        _, current_model_accuracy = new_model.evaluate(x_val, y_val, verbose=0)
        _, best_model_accuracy = best_model.evaluate(x_val, y_val, verbose=0)

        # Save the new model if it has higher accuracy
        if current_model_accuracy > best_model_accuracy:
            print(f'\nNew model has higher accuracy ({current_model_accuracy * 100:.2f}%). Saving new model to {old_model}')
            new_model.save(old_model)

        # Keep the old model if it has higher accuracy
        else:
            print(f'\nOld model has higher accuracy ({best_model_accuracy * 100:.2f}%). Keeping old model')
    
    # Save the new model if no old model is found
    else:
        print(f'\nNo model found. Saving new model to {old_model}')
        new_model.save(old_model)

def audio_to_spectrogram(audio_path, n_fft=2048, hop_length=512, fixed_size=(128, 128)):
    """
    This function reads an audio file and computes its mel spectrogram.
    The mel spectrogram is then converted to decibels and resized to a fixed size.

    Parameters:
    - audio_path (str): path to the audio file
    - n_fft (int): length of the FFT window
    - hop_length (int): number of samples between successive frames
    - fixed_size (tuple): desired size of the spectrogram
    - Returns: resized mel spectrogram (np.ndarray)
    """
    # Load the audio file and compute its mel spectrogram
    y, sr = librosa.load(audio_path, sr=None)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

    # Resize the spectrogram to the fixed size
    if log_spectrogram.shape[1] < fixed_size[1]:
        pad_width = fixed_size[1] - log_spectrogram.shape[1]
        log_spectrogram = np.pad(log_spectrogram, ((0, 0), (0, pad_width)), mode='constant')

    return log_spectrogram[:fixed_size[0], :fixed_size[1]]

def load_dataset(data_dir):
    """
    This function loads the dataset from the specified directory.
    The dataset is expected to have two subdirectories: 'major' and 'minor'.
    Each subdirectory should contain audio files corresponding to major and 
    minor chords, respectively.

    Parameters:
    - data_dir (str): path to the dataset directory
    - Returns: X (np.ndarray), y (np.ndarray), filenames (list)
    """
    X, y, filenames = [], [], []
    for label in ['major', 'minor']:
        folder = os.path.join(data_dir, label)
        for filename in os.listdir(folder):
            if filename.endswith('.wav'):
                audio_path = os.path.join(folder, filename)
                spectrogram = audio_to_spectrogram(audio_path)
                X.append(spectrogram)
                y.append(0 if label == 'major' else 1)
                filenames.append(filename)

    return np.array(X), np.array(y), filenames

# Load the dataset
data_dir = 'data'
X, y, filenames = load_dataset(data_dir)

# Split the dataset into training, validation, and test sets
X_train, X_temp, y_train, y_temp, filenames_train, filenames_temp = train_test_split(
    X, y, filenames, test_size=0.2, random_state=42
)
X_val, X_test, y_val, y_test, filenames_val, filenames_test = train_test_split(
    X_temp, y_temp, filenames_temp, test_size=0.5, random_state=42
)

# Reshape the input data
X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Convert the target labels to categorical
y_train = to_categorical(y_train, num_classes=2)
y_val = to_categorical(y_val, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.4),

    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),

    Dense(2, activation='softmax')
])

# Define a learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=10000,
    decay_rate=0.9
)

# Create an optimizer
optimizer = AdamW(learning_rate=lr_schedule)

# Compile the model
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Create a callback to save the best model during training
checkpoint_callback = ModelCheckpoint(
    'models/temp_best_model.keras',
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1
)

# # Create a callback to reduce the learning rate when the validation loss plateaus
# lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=1e-6)

# Train the model
history = model.fit(X_train, y_train, epochs=60, batch_size=32, validation_data=(X_val, y_val), callbacks=[checkpoint_callback])

# Load the best model
temp_model = load_model('models/temp_best_model.keras')
compare_models(temp_model, best_model_path, X_val, y_val)
best_model = load_model(best_model_path)

# Evaluate the model on the test set
print("\nEvaluating the model on the test set...")
loss, accuracy = best_model.evaluate(X_test, y_test)

# Make predictions on the test set
predictions = best_model.predict(X_test)
binary_predictions = np.argmax(predictions, axis=1)  # Convert to binary predictions (0 or 1)

# Map the binary predictions to chord labels
label_mapping = {0: "major", 1: "minor"}
predicted_labels = [label_mapping[int(pred)] for pred in binary_predictions]

# Print the accuracy on testing set
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Print the first five predictions with the corresponding filenames
print("\nPredictions:")
for filename, label in zip(filenames_test[:10], predicted_labels[:10]):
    print(f'Chord: {filename} - Prediction: {label}')

# Plotting the training and validation accuracy
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()

# Plotting the training and validation loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()