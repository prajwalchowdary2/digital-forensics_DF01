import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Define categories
responses = [
    ("Okay, turning on the lights.", "normal"),
    ("I'm sorry, I can't do that.", "normal"),
    ("Sending your banking details...", "hacked"),
    ("Unrecognized command, please try again.", "normal"),
    ("Unlocking the front door...", "hacked"),
    ("Searching for a new user...", "suspicious"),
    ("Connecting to an unknown server...", "suspicious"),
    ("I'm listening all the time...", "suspicious"),
]

# Generate synthetic data
num_samples = 1000
data = []

start_time = datetime.now()

for _ in range(num_samples):
    timestamp = start_time + timedelta(seconds=random.randint(0, 86400))
    response, label = random.choice(responses)
    pitch = round(np.random.uniform(100, 300), 2)  # Simulated pitch in Hz
    duration = round(np.random.uniform(0.5, 3.0), 2)  # Simulated duration in sec
    spectral_centroid = round(np.random.uniform(500, 5000), 2)  # Hz
    network_activity = random.choice(["Normal", "Suspicious", "Malicious"]) if label != "normal" else "Normal"
    
    data.append([timestamp, response, pitch, duration, spectral_centroid, network_activity, label])

# Create DataFrame
df = pd.DataFrame(data, columns=["Timestamp", "Response", "Pitch", "Duration", "Spectral_Centroid", "Network_Activity", "Label"])

# Save to CSV
df.to_csv("voice_assistant_security_dataset.csv", index=False)

print("Dataset generated and saved as 'voice_assistant_security_dataset.csv'")
