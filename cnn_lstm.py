import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, BatchNormalization, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load dataset
df = pd.read_csv("voice_assistant_security_dataset.csv")

# Selecting features and labels
X = df[["Pitch", "Duration", "Spectral_Centroid"]].values
y = df["Label"].values

# Encode labels (Normal=0, Suspicious=1, Hacked=2)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Scale features
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Reshape X for LSTM (samples, timesteps=1, features=3)
X = np.reshape(X, (X.shape[0], 1, X.shape[1]))  # Ensure X is 3D

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print shape to verify
print(f"X_train shape: {X_train.shape}")

# Build improved CNN-LSTM model
model = Sequential([
    Input(shape=(1, X_train.shape[2])),  # Correct input shape
    Conv1D(128, kernel_size=1, activation='relu', padding='same'),
    BatchNormalization(),
    LSTM(128, return_sequences=True),
    LSTM(64),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')  # 3 classes: Normal, Suspicious, Hacked
])

# Compile model with lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), 
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model for more epochs
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Evaluate model
loss, acc = model.evaluate(X_test, y_test)
print(f"Improved Model Accuracy: {acc:.2f}")

# Save model
model.save("voice_assistant_audio_model.keras")
