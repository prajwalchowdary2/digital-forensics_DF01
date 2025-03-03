import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("home_automation_security.csv")

# Fix column names (remove spaces)
df.columns = df.columns.str.strip()

# Assign "good" by default
df["Status"] = "good"

# Assign "hacked" to Alexa, "suspicious" to Google Mini, and keep others "good"
df.loc[df["Device"] == "Alexa", "Status"] = "hacked"
df.loc[df["Device"] == "Google Mini", "Status"] = "suspicious"

# If an "Anomalies" column exists, assign based on suspicious activity levels
if "Anomalies" in df.columns:
    df.loc[df["Anomalies"] > 5, "Status"] = "suspicious"
    df.loc[df["Anomalies"] > 10, "Status"] = "hacked"

# Save updated dataset
df.to_csv("home_automation_security_updated.csv", index=False)

# Print sample rows to verify
print(df.head())
