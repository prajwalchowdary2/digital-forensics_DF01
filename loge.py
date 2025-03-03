import tkinter as tk
from tkinter import ttk
import ttkbootstrap as tb
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

# Function to fetch Alexa commands
def fetch_alexa_logs():
    status_label.config(text="Fetching logs... Please wait.", foreground="orange")
    root.update_idletasks()

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)
    driver.get("https://www.amazon.in/alexa-privacy/apd/rvh")

    status_label.config(text="Log in to Amazon, then press 'Fetch Logs'", foreground="blue")
    input("Log in manually, then press Enter here...")  # Wait for manual login

    time.sleep(5)  # Ensure page loads fully

    # Find all logs
    history_items = driver.find_elements(By.CLASS_NAME, "record-info")  
    logs = []

    for item in history_items:
        try:
            command = item.find_element(By.XPATH, "./preceding-sibling::div[contains(@class, 'customer-transcript')]").text.strip()
            date = item.find_elements(By.CLASS_NAME, "item")[0].text.strip()  # Extracts Date
            time_recorded = item.find_elements(By.CLASS_NAME, "item")[1].text.strip()  # Extracts Time
            device = item.find_element(By.CLASS_NAME, "device-name").text.strip()

            timestamp = f"{date} {time_recorded}"
            logs.append((timestamp, command, device))

        except Exception as e:
            print("Skipping an item due to error:", e)
            continue

    driver.quit()
    display_logs(logs)

# Function to display logs in Tkinter
def display_logs(logs):
    for row in tree.get_children():
        tree.delete(row)  # Clear previous logs

    if logs:
        animate_logs(logs, 0)  # Call animation function
        status_label.config(text="Logs Loaded Successfully!", foreground="green")
    else:
        status_label.config(text="No logs found.", foreground="red")

# Animated insertion of logs
def animate_logs(logs, index):
    if index < len(logs):
        tree.insert("", "end", values=logs[index])
        root.after(100, animate_logs, logs, index + 1)  # Delay for animation

# GUI Setup
root = tb.Window(themename="darkly")  # Modern dark theme
root.title("Alexa Voice Command Logger")
root.geometry("750x450")

frame = tb.Frame(root)
frame.pack(pady=10, padx=20, fill="both", expand=True)

# Fetch Logs Button with Hover Effect
fetch_button = tb.Button(frame, text="Fetch Logs", bootstyle="primary", command=fetch_alexa_logs)
fetch_button.pack(pady=10)

status_label = tb.Label(frame, text="Click 'Fetch Logs' to get Alexa commands.", font=("Arial", 12))
status_label.pack(pady=5)

# Table with Scrollbar
columns = ("Timestamp", "Command", "Device")
tree = ttk.Treeview(frame, columns=columns, show="headings", style="Treeview")

tree.heading("Timestamp", text="Timestamp")
tree.heading("Command", text="Command")
tree.heading("Device", text="Device")

# Styling
tree.pack(expand=True, fill="both")
style = ttk.Style()
style.configure("Treeview", font=("Arial", 10), rowheight=30)
style.configure("Treeview.Heading", font=("Arial", 12, "bold"))

root.mainloop()
