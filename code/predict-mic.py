import numpy as np
import librosa
import sounddevice as sd
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import ttk

# Cargar el modelo
model = load_model('models/best_model.keras')
label_mapping = {0: "major", 1: "minor"}

# Parámetros de audio
SAMPLE_RATE = 22050  # Frecuencia de muestreo
DURATION = 3  # Duración de la grabación en segundos
FRAME_SIZE = SAMPLE_RATE * DURATION

def audio_to_spectrogram(audio_data, sr=SAMPLE_RATE, n_fft=2048, hop_length=512, fixed_size=(128, 128)):
    """
    Convierte un fragmento de audio en un espectrograma mel.
    """
    spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_fft=n_fft, hop_length=hop_length)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    
    if log_spectrogram.shape[1] < fixed_size[1]:
        pad_width = fixed_size[1] - log_spectrogram.shape[1]
        log_spectrogram = np.pad(log_spectrogram, ((0, 0), (0, pad_width)), mode='constant')
    
    return log_spectrogram[:fixed_size[0], :fixed_size[1]]

def make_prediction(audio_data):
    """
    Realiza la predicción del acorde basado en un fragmento de audio.
    """
    spectrogram = audio_to_spectrogram(audio_data)
    spectrogram = spectrogram[..., np.newaxis]
    spectrogram = np.expand_dims(spectrogram, axis=0)
    prediction = model.predict(spectrogram)
    predicted_label = label_mapping[np.argmax(prediction)]
    return predicted_label

def record_and_predict():
    """
    Graba un fragmento de audio y muestra la predicción.
    """
    print("Grabando...")
    audio_data = sd.rec(int(FRAME_SIZE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()  # Espera a que termine la grabación
    print("Grabación finalizada.")

    audio_data = audio_data.flatten()  # Aplanar los datos de audio
    prediction_label = make_prediction(audio_data)
    
    # Actualizar la etiqueta de la interfaz con la predicción
    prediction_label_var.set(f"Predicción: {prediction_label}")

# Crear la interfaz con Tkinter
window = tk.Tk()
window.title("Clasificador de Acordes en Vivo")

# Etiqueta para mostrar la predicción
prediction_label_var = tk.StringVar()
prediction_label = tk.Label(window, textvariable=prediction_label_var, font=("Helvetica", 16))
prediction_label.pack(pady=20)

# Botón para grabar y predecir
record_button = tk.Button(window, text="Grabar y Predecir", command=record_and_predict)
record_button.pack(pady=20)

window.mainloop()
