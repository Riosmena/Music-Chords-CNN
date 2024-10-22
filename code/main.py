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
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

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
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')  # Outputs:  "major" or "minor"
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=30, validation_data=(X_val, y_val))

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy:.2f}')

# Make predictions on the test set
predictions = model.predict(X_test)
binary_predictions = np.argmax(predictions, axis=1)  # Convert to binary predictions (0 or 1)

# Map the binary predictions to chord labels
label_mapping = {0: "major", 1: "minor"}
predicted_labels = [label_mapping[int(pred)] for pred in binary_predictions]

# Print the first five predictions with the corresponding filenames
print("\nPredictions:")
for filename, label in zip(filenames_test[:5], predicted_labels[:5]):
    print(f'Chord: {filename} - Prediction: {label}')