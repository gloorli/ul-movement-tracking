import h5py
import pandas as pd
import os
from pathlib import Path
from scipy.signal import butter, filtfilt
import numpy as np


def get_file_path():
    """
    Prompts the user to enter a folder path and returns the path to the .mat file in the folder.

    Returns:
        str: The file path to the .mat file.
    """
    # Get folder path input from user
    while True:
        folder_path = 'IMU_data_GT' #input("Enter folder path: ")
        if os.path.exists(folder_path):
            break
        print("Error: folder path does not exist")

    # Get list of .mat files in folder
    mat_files = [file for file in os.listdir(folder_path) if file.endswith('.mat')]

    # Check if there is only one .mat file in folder
    if len(mat_files) == 1:
        file_name = mat_files[0]
    else:
        print("Error: folder should contain only one .mat file")
        exit()

    # Construct the full path to the .mat file
    file_path = os.path.join(folder_path, file_name)
    return file_path

    
def extract_header_data_from_mat_file(file_path):
    """
    Extracts header data from a .mat file and returns a pandas DataFrame.

    Args:
        file_path (str): The path to the .mat file.

    Returns:
        pandas.DataFrame: A DataFrame containing the header data.
    """
    # Open .mat file with h5py
    f = h5py.File(file_path, 'r')
    
    # Access the data you want to put into a pandas DataFrame
    dataset = f['jumpExp']
    
    # Extract header data
    header_data = {}
    for key in dataset['header'].keys():
        value = dataset['header'][key][()]
        if key in ['start', 'stop', 'freq']:
            value = float(value)
        else:
            value = value.tobytes().decode('ascii')
        header_data[str(key)] = value
    
    # Convert header data to a DataFrame
    df_header = pd.DataFrame([header_data])
    
    return df_header


def extract_time_data_from_mat_file(file_path):
    """
    Extracts time data from a .mat file and returns it as a pandas DataFrame.

    Args:
        file_path (str): The path to the .mat file.

    Returns:
        pandas.DataFrame: A DataFrame with the time data.
    """

    # Open .mat file with h5py
    f = h5py.File(file_path, 'r')

    # Access the data you want to put into a pandas DataFrame
    dataset = f['jumpExp']

    # Extract time data
    time_data = dataset['time'][:]
    df_time = pd.DataFrame(time_data, columns=['time'])
    
    return df_time


def extract_sensors_data_from_mat_file(file_path: str, sensor_placement, sensors_features, dataframe_features) -> dict:
    """
    Extracts data from a .mat file and returns a dictionary of dataframes.

    Args:
        file_path (str): The path to the .mat file.

    Returns:
        dict: A dictionary of dataframes, where the keys are the sensor placements
            ('chest', 'LW', 'RW') and the values are dataframes with the sensor data.
    """
    dfs = {placement: pd.DataFrame(columns=dataframe_features) for placement in sensor_placement}
    
    with h5py.File(file_path, 'r') as mat_file:
        for placement_idx, placement in enumerate(sensor_placement):
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

def extract_sensors_data_from_axivity_file(file_path: str, dataframe_features):
    """
    Extracts data from a CSV file and returns a DataFrame.

    Args:
        file_path (str): The path to the CSV file.
        dataframe_features (list): List of desired features in the output dataframe.

    Returns:
        dict: A DataFrame containing a dataframe for the sensor placement 
            present in the CSV.
    """

    # Read data from CSV
    df = pd.DataFrame(columns=dataframe_features)
    df = pd.read_csv(file_path, names=dataframe_features, header=0)

    return df


def export_to_csv(file_path, dfs, sensor_placement, sensors_features, dataframe_features):
    """
    Combines data from multiple dataframes and exports them to a CSV file.

    Args:
        file_path (str): The path to the output CSV file.
        dfs (dict): A dictionary of dataframes, where the keys are the sensor placements
            ('chest', 'LW', 'RW') and the values are dataframes with the sensor data.
    """
    # Create an empty dataframe to hold the combined data
    combined_df = pd.DataFrame()

    # Loop through each sensor placement and add its data to the combined dataframe
    for placement in sensor_placement:
        df = dfs[placement]
        if placement == 'LW':
            prefix = 'wrist_l'
        elif placement == 'RW':
            prefix = 'wrist_r'
        else:
            prefix = 'chest'
        df.columns = [f"{prefix}__{feature}" for feature in dataframe_features]
        combined_df = pd.concat([combined_df, df], axis=1)

    # Export the combined data to a CSV file
    output_dir = os.path.join(os.path.dirname(file_path), 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(file_path))[0] + '.csv')
    combined_df.to_csv(output_file, index=False)


def get_sampling_freq(header):
    return header['freq'].astype('float64').item()


def get_recording_time(time):
    return int(time.iloc[-1])


def filter_gravity(data):
    # Define the filter parameters
    sampling_freq = 50  # Assuming the data is uniformly sampled
    cutoff_freq = 0.25  # Cutoff frequency in Hz
    order = 1  # Filter order

    # Calculate the normalized cutoff frequency
    normalized_cutoff_freq = cutoff_freq / (sampling_freq / 2)

    # Create the high-pass filter coefficients using Butterworth filter
    b, a = butter(order, normalized_cutoff_freq, btype='high', analog=False, output='ba')

    # Extract the acceleration columns to filter
    columns_to_filter = ['acc_x', 'acc_y', 'acc_z']

    # Filter the data using filtfilt to avoid phase distortion
    filtered_data = data.copy()
    for column in columns_to_filter:
        filtered_data[column] = filtfilt(b, a, data[column])

    return filtered_data