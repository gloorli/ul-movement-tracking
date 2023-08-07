import os
import pandas as pd
import numpy as np 
from scipy.interpolate import CubicSpline
from activity_count_function import *
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from utilities import plot_resampled_arrays
from utilities import *


def load_arrays_from_csv(file_path):
    """
    Load six arrays from a CSV file with headers.

    Args:
        file_path (str): File path of the CSV file.

    Returns:
        (tuple): A tuple containing the following arrays:
            - pitch_mad_ndh (numpy.ndarray): NumPy array containing pitch_mad values for the left bronchus.
            - yaw_mad_ndh (numpy.ndarray): NumPy array containing yaw_mad values for the left bronchus.
            - pitch_mad_dh (numpy.ndarray): NumPy array containing pitch_mad values for the right bronchus.
            - yaw_mad_dh (numpy.ndarray): NumPy array containing yaw_mad values for the right bronchus.
            - GT_mask_50Hz_ndh (numpy.ndarray): NumPy array containing GT_mask_50Hz values for the left bronchus.
            - GT_mask_50Hz_dh (numpy.ndarray): NumPy array containing GT_mask_50Hz values for the right bronchus.
    """
    if not os.path.exists(file_path):
        raise ValueError(f"The file '{file_path}' does not exist.")

    # Read the CSV file into a DataFrame
    data_df = pd.read_csv(file_path)

    # Extract the arrays from the DataFrame based on headers
    pitch_mad_ndh = data_df['Pitch MAD NDH'].values
    yaw_mad_ndh = data_df['Yaw MAD NDH'].values
    GT_mask_50Hz_ndh = data_df['GT Mask 50Hz NDH'].values
    pitch_mad_dh = data_df['Pitch MAD DH'].values
    yaw_mad_dh = data_df['Yaw MAD DH'].values
    GT_mask_50Hz_dh = data_df['GT Mask 50Hz DH'].values

    return pitch_mad_ndh, yaw_mad_ndh, pitch_mad_dh, yaw_mad_dh, GT_mask_50Hz_ndh, GT_mask_50Hz_dh


def get_group_data_gm(csv_files_dict):
    """
    Merge arrays per participant from multiple CSV files.

    Args:
        csv_files_dict (dict): Dictionary containing CSV files grouped per participant.

    Returns:
        (tuple): A tuple containing the following lists of arrays (distinct lists for each participant):
            - pitch_mad_ndh (List[numpy.ndarray]): List of NumPy arrays containing pitch_mad values for all participants in NDH.
            - yaw_mad_ndh (List[numpy.ndarray]): List of NumPy arrays containing yaw_mad values for all participants in NDH.
            - pitch_mad_dh (List[numpy.ndarray]): List of NumPy arrays containing pitch_mad values for all participants in DH.
            - yaw_mad_dh (List[numpy.ndarray]): List of NumPy arrays containing yaw_mad values for all participants in DH.
            - GT_mask_50Hz_ndh (List[numpy.ndarray]): List of NumPy arrays containing GT_mask_50Hz values for all participants in NDH.
            - GT_mask_50Hz_dh (List[numpy.ndarray]): List of NumPy arrays containing GT_mask_50Hz values for all participants in DH.
    """
    # Initialize lists of lists to store arrays per participant
    all_pitch_mad_ndh = []
    all_yaw_mad_ndh = []
    all_pitch_mad_dh = []
    all_yaw_mad_dh = []
    all_GT_mask_50Hz_ndh = []
    all_GT_mask_50Hz_dh = []

    # Iterate through the CSV files grouped per participant
    for group_key, group_files in csv_files_dict.items():
        for file_path in group_files:
            # Load the arrays from the CSV file for the current participant
            pitch_mad_ndh, yaw_mad_ndh, pitch_mad_dh, yaw_mad_dh, GT_mask_50Hz_ndh, GT_mask_50Hz_dh = load_arrays_from_csv(file_path)
            
            # Downsample to the smallest dataset 
            
            # Append the arrays to the corresponding lists
            all_pitch_mad_ndh.append(pitch_mad_ndh)
            all_yaw_mad_ndh.append(yaw_mad_ndh)
            all_pitch_mad_dh.append(pitch_mad_dh)
            all_yaw_mad_dh.append(yaw_mad_dh)
            all_GT_mask_50Hz_ndh.append(GT_mask_50Hz_ndh)
            all_GT_mask_50Hz_dh.append(GT_mask_50Hz_dh)

    return all_pitch_mad_ndh, all_yaw_mad_ndh, all_pitch_mad_dh, all_yaw_mad_dh, all_GT_mask_50Hz_ndh, all_GT_mask_50Hz_dh


def resample_angle_data(angle_data, original_frequency, desired_frequency):
    """
    Resamples angle data using cubic spline interpolation.

    Args:
        angle_data (numpy.ndarray): 1D numpy array containing angle values.
        original_frequency (int): Original frequency of angle data.
        desired_frequency (int): Desired frequency for resampling.

    Returns:
        numpy.ndarray: Resampled angle data as a 1D numpy array.
    """
    # Calculate the time array for the original data
    time_original = np.linspace(0, (original_frequency - 1) / original_frequency, original_frequency)

    # Calculate the time array for the desired resampled data
    time_desired = np.linspace(0, (desired_frequency - 1) / desired_frequency, desired_frequency)

    # Create a cubic spline interpolation function
    cubic_spline = CubicSpline(time_original, angle_data)

    # Evaluate the cubic spline at the desired time points to get the resampled data
    resampled_data = cubic_spline(time_desired)

    return resampled_data


def merge_group_data_gm(data_list, mask=True):
    """
    Resamples data arrays from a list to match the smallest length among the arrays.

    Args:
        data_list (list): List of data arrays per participant.
        mask (bool): Boolean flag indicating whether the data is a mask.

    Returns:
        np.ndarray: NumPy array of resampled data arrays.
    """
    all_data = []
    min_length = float('inf')
    elements_removed = []  # List to store the number of elements removed for each array
    resampled_data = []

    for data in data_list:
        # Update the minimum length
        min_length = min(min_length, len(data))
        all_data.append(data)

    for data in all_data:
        original_frequency = len(data)
        desired_frequency = min_length
        
        if mask: 
            # Downsample the binary mask 
            if desired_frequency != original_frequency:
                resampled_values = resample_binary_mask(data, original_frequency, desired_frequency)
            else: 
                resampled_values = data

            # Save the new resampled values inside the resampled_data array 
            resampled_data.append(resampled_values)

            # Plot the data before and after resampling 
            plot_resampled_arrays(data, original_frequency, resampled_values, desired_frequency)

            # Print the number of elements removed by this resampling operation 
            num_removed = len(data) - len(resampled_values)
            elements_removed.append(num_removed)
            
        else:
            # Case of angle resampling
            if desired_frequency != original_frequency:
                resampled_values = resample_angle_data(data, original_frequency, desired_frequency)  # Replace this line with your actual resampling logic for angles
            else: 
                resampled_values = data

            # Save the new resampled values inside the resampled_data array 
            resampled_data.append(resampled_values)

            # Plot the data before and after resampling 
            plot_resampled_arrays(data, original_frequency, resampled_values, desired_frequency)

            # Print the number of elements removed by this resampling operation 
            num_removed = len(data) - len(resampled_values)
            elements_removed.append(num_removed)

    group_data = np.concatenate(resampled_data, axis=0)

    # Print the number of elements removed for each array
    for idx, num_removed in enumerate(elements_removed, start=1):
        print(f"Elements removed in array {idx}: {num_removed}")

    return group_data


def window_data(data, original_sampling_frequency, window_length_seconds):
    """
    Window the data into subarrays of the desired epoch length.

    Args:
        data (numpy.ndarray): 1D numpy array of data.
        original_sampling_frequency (int): Original sampling frequency of the data.
        window_length_seconds (float): Desired epoch length in seconds.

    Returns:
        List of numpy.ndarray: List of subarrays containing data per window.
    """
    # Calculate the number of data points per window
    window_length = int(window_length_seconds * original_sampling_frequency)

    # Calculate the number of windows that can be created
    num_windows = len(data) // window_length

    # Initialize the list to store the windowed data
    windowed_data = []

    # Extract data per window and store it in the windowed_data list
    for i in range(num_windows):
        start_idx = i * window_length
        end_idx = start_idx + window_length
        window_data = data[start_idx:end_idx]
        windowed_data.append(window_data)

    return windowed_data


def compute_GMAC(pitch_mad_50Hz, AC_1Hz, ac_threshold = 0, functional_space = 30):

    # Compute windows of 1 seconds of data for the pitch angles
    theta_per_epoch = window_data(pitch_mad_50Hz, original_sampling_frequency = 50, window_length_seconds = 1)
    
    num_epoch_pitch = len(theta_per_epoch)
    
    # Initialize an array to store the mean pitch per epoch
    theta_mean_per_epoch = np.zeros(num_epoch_pitch)
    
    # Iterate over each epoch and compute the mean pitch angle
    for i in range(num_epoch_pitch):
        theta_mean_per_epoch[i] = np.mean(theta_per_epoch[i])
    
    # Ensure AC and mean pitch per epoch have same size 
    theta_mean_per_epoch, AC_1Hz = remove_extra_elements(theta_mean_per_epoch, AC_1Hz)
    
    # Number of epochs GMAC 
    num_epoch_gmac = len(theta_mean_per_epoch)
    
    # Initialize an array to store GMAC score @ 1Hz
    GMAC = np.zeros(num_epoch_gmac)
    
    for i in range(num_epoch_gmac):
        if ((np.abs(theta_mean_per_epoch[i]) < functional_space) and (AC_1Hz[i] > ac_threshold)):
            GMAC[i] = 1
    return GMAC 