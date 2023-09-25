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
from imu_video_synch import *


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

    plt.figure(figsize=(20, 6))  # Modified figure size for subplots

    
    global_y_max = max(
    metric_duration['ndh']['conv']['percentage_active'],
    metric_duration['ndh']['opt']['percentage_active'],
    metric_duration['dh']['conv']['percentage_active'],
    metric_duration['dh']['opt']['percentage_active']
    )

    sides = ['DH', 'NDH']
    side_plots = ['Non-Aff. H', 'Aff. H' ]
    
    

    for i, (side, side_plot) in enumerate(zip(sides, side_plots)):
        plt.subplot(1, 2, i + 1)  # 1 row, 2 columns, index i+1
        plt.title(f"Functional Arm Use Duration Comparison - {side_plot} - {metric_name}", fontsize=18, pad=20)


        metric_duration_conv_percentage = metric_duration[side.lower()]['conv']['percentage_active']
        metric_duration_opt_percentage = metric_duration[side.lower()]['opt']['percentage_active']
        
         # Check for division by zero and handle accordingly
        if metric_duration_conv_percentage == 0:
            diff_opt = float('inf')  # Assign infinity if the denominator is zero
            diff_opt_symbol = '+'  # In this case, the percentage increase is theoretically infinite
        else:
            diff_opt = ((metric_duration_opt_percentage - metric_duration_conv_percentage) / metric_duration_conv_percentage) * 100
            diff_opt_symbol = '+' if diff_opt >= 0 else '-'
            

        x = [0, 1]
        heights = [metric_duration_conv_percentage, metric_duration_opt_percentage]
        labels = ['Conventional', 'Optimal']
        colors = ['#007BFF', '#28a745']

        ax = sns.barplot(x=x, y=heights, palette=colors, edgecolor=".2")
        plt.xticks(x, labels)
        plt.ylabel('Percentage Active Duration (%)', fontsize=14)
        plt.xlabel('')

        durations = [
            metric_duration[side.lower()]['conv']['duration_formatted'],
            metric_duration[side.lower()]['opt']['duration_formatted']
        ]

        for j, duration in enumerate(durations):
            ax.text(j, heights[j] / 2, duration, ha='center', fontsize=16, fontweight='bold')

        y_max = max(metric_duration_conv_percentage, metric_duration_opt_percentage)
      
        plt.ylim(0, global_y_max + global_y_max * 0.2)


        diff_opt = ((metric_duration_opt_percentage - metric_duration_conv_percentage) / metric_duration_conv_percentage) * 100
        diff_opt_symbol = '+' if diff_opt >= 0 else '-'
        
        # Handle infinity when displaying the text
        if diff_opt == float('inf'):
            ax.text(1, metric_duration_opt_percentage + y_max * 0.01, "Deviation from reference: ∞%", ha='center', fontsize=12, fontweight='bold', color='black')
        else:
            ax.text(1, metric_duration_opt_percentage + y_max * 0.01, f"Deviation from reference: {diff_opt_symbol}{abs(int(diff_opt))}%", ha='center', fontsize=12, fontweight='bold', color='black')
        
        
        #ax.text(0, metric_duration_conv_percentage + y_max * 0.01, "Reference", ha='center', fontsize=12, fontweight='bold', color='black')
        #ax.text(1, metric_duration_opt_percentage + y_max * 0.01, f"Deviation from reference: {diff_opt_symbol}{abs(int(diff_opt))}%", ha='center', fontsize=12, fontweight='bold', color='black')

    if save_path:
        file_name = f"Functional_Arm_Use_Duration_Comparison_{metric_name}.png"
        full_file_path = os.path.join(save_path, file_name)
        plt.savefig(full_file_path)
        print(f"Figure saved as '{full_file_path}'")
        plt.show()
    else:
        plt.show()


def get_data_48h_imu(participant_path, imu_path, start_date, end_date, affected_hand,
                     sensor_placement, matlab_sensor_features, sensor_features, num_sensors,
                     frequency_imu):

    try:
        # Read existing Feather files if available
        def load_feather_data(participant_path, data_type):
            path_to_data = os.path.join(participant_path, f"trimmed_{data_type}_data.feather")
            return pd.read_feather(path_to_data)
        
        trimmed_RW_data = load_feather_data(participant_path, 'RW')
        trimmed_LW_data = load_feather_data(participant_path, 'LW')
        if num_sensors == 3:
            trimmed_chest_data = load_feather_data(participant_path, 'chest')
        
        print("Successfully loaded the Feather files into DataFrames.")

    except FileNotFoundError:
        # Perform full data extraction if Feather files are not available
        # Full extraction of data using the mat file 
        print('Feather files not found: Extraction using IMU mat files')
        header = extract_header_data_from_mat_file(imu_path)
        sampling_freq =get_sampling_freq(header)
        time_array = extract_time_data_from_mat_file(imu_path)
        recording_time = get_recording_time(time_array)
        dfs = extract_sensors_data_from_mat_file_48h(imu_path, sensor_placement, matlab_sensor_features, sensor_features, num_sensors = num_sensors)
        acc_LW = dfs['LW'][['acc_x', 'acc_y', 'acc_z']]
        acc_RW = dfs['RW'][['acc_x', 'acc_y', 'acc_z']]
        gyro_LW = dfs['LW'][['gyro_x', 'gyro_y', 'gyro_z']]
        gyro_RW = dfs['RW'][['gyro_x', 'gyro_y', 'gyro_z']]
        mag_LW = dfs['LW'][['magneto_x', 'magneto_y', 'magneto_z']]
        quat_sensor_LW = dfs['LW'][['quat_0', 'quat_1', 'quat_2','quat_3']]
        LW_data = dfs['LW']
        RW_data = dfs['RW']
        if num_sensors == 3:
            chest_data = dfs['chest']

        # Add timestamps to datasets 
        IMU_start_timestamp, IMU_end_timestamp = get_datetime_timestamp(header)
        timestamps_array = create_timestamps(IMU_start_timestamp, IMU_end_timestamp, frequency_imu)
        # Add timestamps to raw data
        LW_data = pd.concat([timestamps_array, LW_data], axis=1)
        RW_data = pd.concat([timestamps_array, RW_data], axis=1)
        if num_sensors == 3:
            chest_data = pd.concat([timestamps_array, chest_data], axis=1)

         # Check if start_date and end_date are provided
        if start_date is not None and end_date is not None:
            trimmed_LW_data = trim_dataframe_by_time(LW_data, start_date, end_date)
            trimmed_RW_data = trim_dataframe_by_time(RW_data, start_date, end_date)
            if num_sensors == 3:
                trimmed_chest_data = trim_dataframe_by_time(chest_data, start_date, end_date)
        else:
            print("No start and end dates provided. Not trimming the data.")
            trimmed_LW_data = LW_data
            trimmed_RW_data = RW_data
            if num_sensors == 3:
                trimmed_chest_data = chest_data

        # Saving the dataset as feather files, easier to load and manage for large dataset
        # Save the trimmed dataset to avoid having to redo the trimming operation 
        # Use Feather package to be more efficient   
        save_dataframe_feather(trimmed_LW_data, participant_path, "trimmed_LW_data")
        save_dataframe_feather(trimmed_RW_data, participant_path, "trimmed_RW_data")
        if num_sensors == 3:
            save_dataframe_feather(trimmed_chest_data, participant_path, "trimmed_chest_data")

    # Non Dominant and Dominant Hand dataset attribution
    ndh_data = trimmed_RW_data if affected_hand.lower() == 'right' else trimmed_LW_data
    dh_data = trimmed_LW_data if affected_hand.lower() == 'right' else trimmed_RW_data

    if num_sensors == 3:
        return ndh_data, dh_data, trimmed_chest_data
    else: 
        return ndh_data, dh_data





