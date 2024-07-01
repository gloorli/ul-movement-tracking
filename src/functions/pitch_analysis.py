import pandas as pd
import numpy as np

def read_pitch_from_csv(file_path):
    """
    Read pitch data from a CSV file.
    """
    pitch_data = pd.read_csv(file_path)
    pitch_data = pitch_data[['motionPitch(rad)', 'loggingTime(txt)']]
    pitch_data['motionPitch(deg)'] = pitch_data['motionPitch(rad)'].apply(lambda x: x * 180 / np.pi)
    return pitch_data

def plot_pitch_data(pitch_data):
    """
    Plot pitch data.
    """
    pitch_data.plot(x='loggingTime(txt)', y='motionPitch(deg)', title='Pitch data', figsize=(12, 6))