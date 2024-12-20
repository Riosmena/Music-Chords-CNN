import os
import numpy as np
import librosa
import sounddevice as sd
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

# Load the model
model = load_model('models/best_model.keras')
label_mapping = {0: "major", 1: "minor"}

# Convert audio to spectrogram
def audio_to_spectrogram(audio_path, n_fft=2048, hop_length=512, fixed_size=(128, 128)):
    y, sr = librosa.load(audio_path, sr=None)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

    if log_spectrogram.shape[1] < fixed_size[1]:
        pad_width = fixed_size[1] - log_spectrogram.shape[1]
        log_spectrogram = np.pad(log_spectrogram, ((0, 0), (0, pad_width)), mode='constant')
    
    return log_spectrogram[:fixed_size[0], :fixed_size[1]]

# Make a prediction
def make_prediction(audio_path):
    spectrogram = audio_to_spectrogram(audio_path)
    spectrogram = spectrogram[..., np.newaxis]
    spectrogram = np.expand_dims(spectrogram, axis=0)
    prediction = model.predict(spectrogram)
    predicted_label = label_mapping[np.argmax(prediction)]
    return predicted_label

# Select a directory
def select_directory():
    directory = filedialog.askdirectory()
    if directory:
        audio_files = [f for f in os.listdir(directory) if f.endswith('.wav')]
        for file in audio_files:
            full_path = os.path.join(directory, file)
            pred_label = make_prediction(full_path)
            files_list.insert('', 'end', values=(file, pred_label, full_path))

def play_audio(path):
    y, sr = librosa.load(path, sr=None)
    sd.play(y, sr)

# Create the main window
window = tk.Tk()
window.title("Chord Classifier")
window.geometry("500x400")

# Create a button to select the directory
select_button = tk.Button(window, text="Select audio path", command=select_directory)
select_button.pack(pady=10)

# Create a table to display the predictions
columns = ("File", "Prediction", "Path")
files_list = ttk.Treeview(window, columns=columns, show='headings')
files_list.heading("File", text="File")
files_list.heading("Prediction", text="Prediction")
files_list.heading("Path", text="Path")
files_list.column("Path", width=0, stretch=tk.NO) 
files_list.pack(pady=10, fill="both", expand=True)

# Play button
def on_play_button():
    selected_item = files_list.selection()
    if selected_item:
        path = files_list.item(selected_item[0], 'values')[2]
        play_audio(path)

play_button = tk.Button(window, text="Play selected chord", command=on_play_button)
play_button.pack(pady=10)

window.mainloop()
