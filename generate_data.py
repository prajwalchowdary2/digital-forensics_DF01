import random
import pandas as pd

def generate_live_data():
    """Simulates real-time IoT device activity data."""
    return {
        "device": random.choice(["Alexa", "HomePod"]),
        "cpu_usage": round(random.uniform(10, 90), 2),  # Random CPU usage (10-90%)
        "network_activity": round(random.uniform(1, 1000), 2),  # Network traffic
        "memory_usage": round(random.uniform(100, 8000), 2),  # Memory in MB
        "anomalous_behavior": random.choice([0, 1])  # 0 = Normal, 1 = Suspicious
    }

def generate_dataset(size=100000):
    """Creates a dataset of simulated IoT activity."""
    data = [generate_live_data() for _ in range(size)]
    df = pd.DataFrame(data)
    df.to_csv("device_data.csv", index=False)
    print(f"✅ Dataset saved as 'device_data.csv' with {size} records.")

if __name__ == "__main__":
    generate_dataset(100000)

