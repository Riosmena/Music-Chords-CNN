import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

# Función para convertir un archivo de audio en un espectrograma de tamaño fijo
def audio_to_spectrogram(audio_path, n_fft=2048, hop_length=512, fixed_size=(128, 128)):
    y, sr = librosa.load(audio_path, sr=None)  # Cargar el audio
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)  # Escala logarítmica

    # Asegurarse de que el espectrograma tenga tamaño fijo (128x128)
    if log_spectrogram.shape[1] < fixed_size[1]:  # Padding si es muy pequeño
        pad_width = fixed_size[1] - log_spectrogram.shape[1]
        log_spectrogram = np.pad(log_spectrogram, ((0, 0), (0, pad_width)), mode='constant')
    return log_spectrogram[:fixed_size[0], :fixed_size[1]]  # Cortar si es muy grande

# Recorrer los archivos del dataset y generar espectrogramas
def load_dataset(data_dir):
    X, y = [], []
    for label in ['major', 'minor']:
        folder = os.path.join(data_dir, label)
        for filename in os.listdir(folder):
            if filename.endswith('.wav'):
                audio_path = os.path.join(folder, filename)
                spectrogram = audio_to_spectrogram(audio_path)
                X.append(spectrogram)
                y.append(0 if label == 'major' else 1)  # 0 = Mayor, 1 = Menor
    return np.array(X), np.array(y)

# Cargar el dataset
data_dir = 'data'
X, y = load_dataset(data_dir)

X_resized = np.array([librosa.util.fix_length(x, size=128, axis=1)[:128, :] for x in X])

# Dividir en conjuntos de entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X_resized, y, test_size=0.2, random_state=42)

# Crear el modelo CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Salida binaria: 0 (Mayor), 1 (Menor)
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train[..., np.newaxis], y_train, epochs=10, validation_data=(X_val[..., np.newaxis], y_val))

# Evaluar el modelo
loss, accuracy = model.evaluate(X_val[..., np.newaxis], y_val)
print(f'Precisión en validación: {accuracy:.2f}')

# Realizar predicciones
predictions = model.predict(X_val[..., np.newaxis])

# Convertir las probabilidades a etiquetas con un umbral de 0.5
predictions = (predictions >= 0.5).astype(int)

print(predictions[:10])  # Mostrará ['major', 'major', 'major', 'minor', 'minor']

