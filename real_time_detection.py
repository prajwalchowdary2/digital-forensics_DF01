import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import load_model
from generate_data import generate_live_data  # Import live data generator

# Load model & scaler
model = load_model("anomaly_detection_model.h5")
scaler_mean = np.load("scaler.npy")
scaler_var = np.load("scaler_var.npy")

# Load label encoder
label_encoder = LabelEncoder()
label_encoder.fit(["Alexa", "HomePod"])  # Ensure same encoding as training

# Simulate real-time detection
def detect_anomaly():
    """Simulates real-time device monitoring and classifies as safe or hacked."""
    
    # Generate live data for detection
    live_data = generate_live_data()
    df = pd.DataFrame([live_data])

    # Encode 'device' column
    df["device"] = label_encoder.transform(df["device"])

    # Separate out the 'device' column (categorical) and the numerical features
    device_column = df["device"].copy()  # Save the 'device' column for later
    df = df.drop("device", axis=1)  # Drop the 'device' column before scaling

    # Initialize the scaler
    scaler = StandardScaler()
    scaler.mean_ = scaler_mean
    scaler.var_ = scaler_var
    scaler.scale_ = np.sqrt(scaler.var_)  # Correctly set the scale (std dev)

    # Normalize features (only numerical data)
    df_scaled = scaler.transform(df)

    # Now pass only the numerical features (without the 'device' column) into the model
    # Predict anomaly using the pre-trained model
    prediction = model.predict(df_scaled)[0][0]
    result = "HACKED ⚠️" if prediction > 0.5 else "SAFE ✅"

    # Output the results
    print(f"🔍 Live Data: {live_data}")
    print(f"📢 Status: {result}")

# Run real-time detection
if __name__ == "__main__":
    detect_anomaly()
