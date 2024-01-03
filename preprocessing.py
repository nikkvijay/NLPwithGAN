import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tkinter as tk
from tkinter import filedialog, messagebox
import os

# Function to open a file dialog and return the selected file path
def select_file():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename()  # Open the file dialog
    return file_path

# Function to generate the path for the preprocessed file
def get_preprocessed_file_path(original_path):
    directory, filename = os.path.split(original_path)
    name, _ = os.path.splitext(filename)
    new_filename = f"{name}_preprocessed.csv"
    return os.path.join(directory, new_filename)

# Selecting dataset file
file_path = select_file()

# Check if the file path is empty (user cancelled) or file format is unsupported
if not file_path or (not file_path.endswith('.csv') and not file_path.endswith('.parquet')):
    messagebox.showwarning("Unsupported File", "Unsupported file format. Please provide a CSV or Parquet file.")
else:
    # Reading the dataset
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)

    # Preprocessing
    df.columns = df.columns.str.strip()
    df.replace(['Infinity', '-Infinity', np.inf, -np.inf], np.nan, inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

    # Encoding 'Label' column
    if 'Label' in df.columns:
        label_encoder = LabelEncoder()
        df['Label'] = label_encoder.fit_transform(df['Label'].astype(str))

    # Normalization
    scaler = MinMaxScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    # Saving the preprocessed file
    preprocessed_file_path = get_preprocessed_file_path(file_path)
    df.to_csv(preprocessed_file_path, index=False)

    # Displaying a message box with the save location
    messagebox.showinfo("File Saved", f"Preprocessed data saved to:\n{preprocessed_file_path}")

# Close the root Tkinter instance
tk.Tk().destroy()
