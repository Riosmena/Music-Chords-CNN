import numpy as np
import librosa
import librosa.display
import sounddevice as sd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load the model
model = load_model('models/best_model.keras')
label_mapping = {0: "major", 1: "minor"}

# Audio parameters
SAMPLE_RATE = 22050
DURATION = 4
FRAME_SIZE = SAMPLE_RATE * DURATION
audio_data = None

def audio_to_spectrogram(audio_data, sr=SAMPLE_RATE, n_fft=2048, hop_length=512, fixed_size=(128, 128)):
    spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_fft=n_fft, hop_length=hop_length)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

    if log_spectrogram.shape[1] < fixed_size[1]:
        pad_width = fixed_size[1] - log_spectrogram.shape[1]
        log_spectrogram = np.pad(log_spectrogram, ((0, 0), (0, pad_width)), mode='constant')
    
    return log_spectrogram[:fixed_size[0], :fixed_size[1]]

def make_prediction(audio_data):
    spectrogram = audio_to_spectrogram(audio_data)
    spectrogram = spectrogram[..., np.newaxis]
    spectrogram = np.expand_dims(spectrogram, axis=0)
    prediction = model.predict(spectrogram)
    predicted_label = label_mapping[np.argmax(prediction)]
    return predicted_label

def record_audio():
    global audio_data
    print("Recording...")
    audio_data = sd.rec(int(FRAME_SIZE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    print("Recording finished.")

    audio_data = audio_data.flatten()

    # Make a prediction and update the label
    prediction_label = make_prediction(audio_data)
    prediction_label_var.set(f"Prediction: {prediction_label}")

    # Display the spectrogram
    display_spectrogram(audio_data)

def display_spectrogram(audio_data):
    for widget in canvas_frame.winfo_children():
        widget.destroy()

    spectrogram = audio_to_spectrogram(audio_data)
    fig, ax = plt.subplots(figsize=(5, 4))
    librosa.display.specshow(spectrogram, sr=SAMPLE_RATE, hop_length=512, x_axis='time', y_axis='mel', ax=ax)
    ax.set(title="Espectrograma Mel del Acorde")
    fig.colorbar(ax.collections[0], ax=ax, format='%+2.0f dB')

    canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

def play_audio():
    if audio_data is not None:
        sd.play(audio_data, SAMPLE_RATE)
    else:
        messagebox.showinfo("Error", "You need to record audio first.")

def stop_audio_and_exit():
    sd.stop() 
    exit(0)

# Create the main window
window = tk.Tk()
window.title("Live Chord Classifier")

# Label to display the prediction
prediction_label_var = tk.StringVar()
prediction_label = tk.Label(window, textvariable=prediction_label_var, font=("Helvetica", 16))
prediction_label.pack(pady=10)

# Button to record audio
record_button = tk.Button(window, text="Record & Predict", command=record_audio)
record_button.pack(pady=10)

# Button to play the recorded audio
play_button = tk.Button(window, text="Play Audio", command=play_audio)
play_button.pack(pady=10)

# Frame to display the spectrogram
canvas_frame = tk.Frame(window)
canvas_frame.pack(pady=10)

# Protocol to stop the audio and exit
window.protocol("WM_DELETE_WINDOW", stop_audio_and_exit)

window.mainloop()