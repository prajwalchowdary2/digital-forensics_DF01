import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import tensorflow as tf

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def generate_dataset(n_samples=10000):
    """
    Generate synthetic device data with anomaly patterns
    """
    # Generate timestamps
    timestamps = pd.date_range(start='2024-01-01', periods=n_samples, freq='5min')
    
    # Initialize dataframe
    df = pd.DataFrame(index=timestamps)
    
    # Generate device types
    devices = ['Alexa', 'HomePod']
    df['device'] = np.random.choice(devices, size=n_samples)
    
    # Generate normal patterns
    df['cpu_usage'] = np.random.normal(30, 5, n_samples)  # Normal CPU usage around 30%
    df['memory_usage'] = np.random.normal(40, 8, n_samples)  # Normal memory usage around 40%
    df['network_traffic'] = np.random.normal(50, 10, n_samples)  # Normal network traffic
    df['response_time'] = np.random.normal(100, 20, n_samples)  # Response time in ms
    
    # Add daily patterns
    time_of_day = df.index.hour + df.index.minute/60
    df['cpu_usage'] += 10 * np.sin(2 * np.pi * time_of_day / 24)
    df['memory_usage'] += 15 * np.sin(2 * np.pi * time_of_day / 24)
    
    # Generate anomalies (10% of data)
    anomaly_idx = np.random.choice(n_samples, size=int(n_samples*0.1), replace=False)
    anomaly_timestamps = df.index[anomaly_idx]
    
    # Different types of anomalies
    for timestamp in anomaly_timestamps:
        anomaly_type = np.random.randint(0, 4)
        
        if anomaly_type == 0:
            # Spike in CPU usage
            df.loc[timestamp, 'cpu_usage'] = np.random.uniform(80, 100)
        elif anomaly_type == 1:
            # Memory leak
            df.loc[timestamp, 'memory_usage'] = np.random.uniform(85, 95)
        elif anomaly_type == 2:
            # Network congestion
            df.loc[timestamp, 'network_traffic'] = np.random.uniform(90, 100)
        else:
            # High response time
            df.loc[timestamp, 'response_time'] = np.random.uniform(400, 500)
    
    # Create anomaly labels
    df['anomalous_behavior'] = 0
    df.loc[anomaly_timestamps, 'anomalous_behavior'] = 1
    
    # Add some noise
    df['cpu_usage'] += np.random.normal(0, 2, n_samples)
    df['memory_usage'] += np.random.normal(0, 3, n_samples)
    df['network_traffic'] += np.random.normal(0, 4, n_samples)
    df['response_time'] += np.random.normal(0, 10, n_samples)
    
    # Ensure values are within reasonable ranges
    df['cpu_usage'] = df['cpu_usage'].clip(0, 100)
    df['memory_usage'] = df['memory_usage'].clip(0, 100)
    df['network_traffic'] = df['network_traffic'].clip(0, 100)
    df['response_time'] = df['response_time'].clip(0, 1000)
    
    return df

# Generate dataset
df = generate_dataset(10000)
