import pyarrow.feather as feather
import os
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
# Standard libraries
import os
import csv
import json
import math
import signal
import inspect
from datetime import datetime
import re
import glob
import seaborn as sns

# Data processing and analysis
import numpy as np
import pandas as pd
import subprocess

# Scientific computing
from scipy.signal import (butter, filtfilt, find_peaks, resample,
                          savgol_filter, freqz, lfilter)
from scipy.stats import spearmanr, circmean
from scipy.spatial.transform import Rotation
from scipy.interpolate import CubicSpline, interp1d
from scipy.ndimage import zoom

# Visualization
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D

# Progress bar
from tqdm import tqdm

from zurich_move_data_extraction import *


def extract_sensors_data_from_mat_file_48h(file_path: str, sensor_placement, sensors_features, dataframe_features, num_sensors=3) -> dict:
    """
    Extracts data from a .mat file and returns a dictionary of dataframes.

    Args:
        file_path (str): The path to the .mat file.
        sensor_placement (list): List of sensor placements ('LW', 'RW', 'chest').
        sensors_features (list): List of sensor features to extract.
        dataframe_features (list): List of column names for the dataframes.
        num_sensors (int): Number of sensors present (2 or 3).

    Returns:
        dict: A dictionary of dataframes, where the keys are the sensor placements
            ('LW', 'RW', 'chest') and the values are dataframes with the sensor data.
    """
    dfs = {placement: pd.DataFrame(columns=dataframe_features) for placement in sensor_placement}
    
    with h5py.File(file_path, 'r') as mat_file:
        for placement_idx, placement in enumerate(sensor_placement):
            if num_sensors == 2 and placement == 'chest':
                # Skip processing the chest sensor if there are only 2 sensors
                continue
            
            if placement_idx >= len(mat_file['jumpExp']['sensors'][sensors_features[0]]):
                # Handle cases where the placement index is out of range
                print(f"Skipping {placement} data due to index out of range.")
                continue

            feature_data = {}
            for feature in sensors_features:
                sensors_data = mat_file['jumpExp']['sensors'][feature]
                ref = sensors_data[placement_idx, 0]
                if feature == 'press':
                    feature_data[feature] = np.array(mat_file[ref]).T
                else:
                    feature_data[feature] = np.array(mat_file[ref])
            data = np.concatenate([feature_data[feature] for feature in sensors_features], axis=0)
            df = pd.DataFrame(data.T, columns=dataframe_features)
            dfs[placement] = df
            
    return dfs


def trim_dataframe_by_time(df, start_time, end_time):
    df['timestamp'] = pd.to_datetime(df['timestamp'])  # Convert 'timestamp' column to datetime format
    mask = (df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)  # Create a boolean mask
    trimmed_df = df[mask]  # Apply the mask to the DataFrame
    return trimmed_df


def save_dataframe_feather(df, path, file_name):
    """
    Save a DataFrame using Feather format.
    
    Parameters:
    - df (pandas.DataFrame): The DataFrame to be saved.
    - path (str): Directory where the DataFrame will be saved.
    - file_name (str): Name of the file, without the extension.
    
    Returns:
    - None
    """
    
    # Create the directory if it does not exist
    if not os.path.exists(path):
        os.makedirs(path)
        
    full_path = os.path.join(path, f"{file_name}.feather")
    
    # Save the DataFrame
    feather.write_feather(df, full_path)
    print(f"DataFrame saved at {full_path}")

    
def load_threshold_from_csv(path):
    """
    Load 'Threshold' data from a CSV file given its path.

    Parameters:
    - path (str): Path to the CSV file.

    Returns:
    - thresholds (array-like): Array of thresholds from the 'Threshold' column.
    """
    
    # Load the CSV file into a DataFrame
    df = pd.read_csv(path)
    
    # Extract the 'Threshold' column
    thresholds = df['Threshold'].values
    
    return thresholds


def load_or_generate_data(initial_path, imu_path):
    """
    Attempt to load trimmed_LW_data and trimmed_RW_data from Feather files.
    If they don't exist, extract and prepare the data from the given imu_path.

    Parameters:
    - initial_path (str): Directory where trimmed datasets are/will be saved.
    - imu_path (str): Path to the IMU data in MATLAB format.

    Returns:
    - trimmed_LW_data (pandas.DataFrame): Loaded or generated DataFrame.
    - trimmed_RW_data (pandas.DataFrame): Loaded or generated DataFrame.
    """
    
    # Define paths for Feather files
    trimmed_LW_data_path = os.path.join(initial_path, "trimmed_LW_data.feather")
    trimmed_RW_data_path = os.path.join(initial_path, "trimmed_RW_data.feather")
    
    # Try loading trimmed data
    if os.path.exists(trimmed_LW_data_path) and os.path.exists(trimmed_RW_data_path):
        trimmed_LW_data = feather.read_feather(trimmed_LW_data_path)
        trimmed_RW_data = feather.read_feather(trimmed_RW_data_path)
        print(f"Successfully loaded trimmed datasets.")
        return trimmed_LW_data, trimmed_RW_data
    
    print(f"Trimmed datasets not found. Extracting and preparing data.")
    
    
    return 0,0


def compare_arm_use_duration_plot_48h(metric_duration, metric_name, save_path=None):
    """
    Compare arm use duration using bar charts only for conventional and optimal approaches.

    Args:
        metric_duration (dict): Metric duration arm use dictionary.
        metric_name (str): Name of the metric being compared ('AC', 'GM', or 'GMAC').
        save_path (str, optional): Path to save the figure. If None, the figure won't be saved.

    Returns:
        None
    """
    sns.set(style="white")
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    
    sides = ['NDH', 'DH', 'BIL']
    
    for side in sides:
        plt.figure(figsize=(10, 6))
        plt.title(f"Functional Arm Use Duration Comparison - {side} - {metric_name}", fontsize=18, pad=20)
        
        metric_duration_conv_percentage = metric_duration[side.lower()]['conv']['percentage_active']
        metric_duration_opt_percentage = metric_duration[side.lower()]['opt']['percentage_active']
        
        x = [0, 1]
        heights = [metric_duration_conv_percentage, metric_duration_opt_percentage]
        labels = ['Conventional', 'Optimal']
        colors = ['#007BFF', '#28a745']  # Slightly modified shades of blue and green for modern look
        
        ax = sns.barplot(x=x, y=heights, palette=colors, edgecolor=".2")
        plt.xticks(x, labels)
        plt.ylabel('Percentage Active Duration (%)', fontsize=14)
        plt.xlabel('')
        
        durations = [
            metric_duration[side.lower()]['conv']['duration_formatted'],
            metric_duration[side.lower()]['opt']['duration_formatted']
        ]
        
        for i, duration in enumerate(durations):
            ax.text(i, heights[i] / 2, duration, ha='center', fontsize=16, fontweight='bold')  # Inside bar plot
        
        # Calculate and display percentage differences
        reference_val = metric_duration_conv_percentage  # Conventional is always the reference
        diff_opt = ((metric_duration_opt_percentage - reference_val) / reference_val) * 100
        diff_opt_symbol = '+' if diff_opt >= 0 else '-'

        # Set y-axis limit to be higher than the highest bar by a margin
        y_max = max(metric_duration_conv_percentage, metric_duration_opt_percentage)
        plt.ylim(0, y_max + y_max * 0.2)  # Add 20% extra space at the top

        # Position text label on top of each bar
        ax.text(0, metric_duration_conv_percentage + y_max * 0.01, "Reference", ha='center', fontsize=12, fontweight='bold', color='black')
        ax.text(1, metric_duration_opt_percentage + y_max * 0.01, f"Deviation from reference: {diff_opt_symbol}{abs(int(diff_opt))}%", ha='center', fontsize=12, fontweight='bold', color='black')
        
        if save_path:
            file_name = f"Functional_Arm_Use_Duration_Comparison_{side}_{metric_name}.png"
            full_file_path = os.path.join(save_path, file_name)
            plt.savefig(full_file_path)
            print(f"Figure saved as '{full_file_path}'")
            plt.show()
        else:
            plt.show()
