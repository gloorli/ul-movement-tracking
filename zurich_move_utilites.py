import pandas as pd
import csv
import numpy as np
from datetime import datetime
from matplotlib.colors import ListedColormap
from ahrs.filters import Mahony, Madgwick
from ahrs.common import orientation
import os
from scipy import signal
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import resample
import math
from scipy.signal import savgol_filter, freqz, lfilter
from scipy import interpolate
from scipy.spatial.transform import Rotation
from scipy.stats import circmean
from scipy.interpolate import CubicSpline
from math import nan
import inspect


def get_statistics(data):
    """
    Calculate various statistics for a given array of data and plot the distribution.

    Args:
        data (list or numpy array): An array of data to compute statistics for.

    Returns:
        dict: A dictionary containing the computed statistics.
    """
    statistics = {}
    statistics['mean'] = np.nanmean(data)
    statistics['median'] = np.nanmedian(data)
    statistics['iqr'] = np.nanpercentile(data, 75) - np.nanpercentile(data, 25)
    statistics['range'] = np.nanmax(data) - np.nanmin(data)
    statistics['std'] = np.nanstd(data)
    statistics['max'] = np.nanmax(data)
    statistics['min'] = np.nanmin(data)
    statistics['num_elements'] = len(data)

    # Plotting the data distribution
    plt.figure(figsize=(10, 6))  # Increase the plot size
    counts, bins, _ = plt.hist(data, bins='auto')
    plt.title('Data Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # Add percentage labels to the bars
    total = len(data)
    for count, bin in zip(counts, bins):
        if count > 0:
            percentage = count / total * 100
            plt.text(bin, count, f'{percentage:.1f}%', ha='center', va='bottom')

    plt.show()

    return statistics


def plot_raw_data(data, sampling_freq):
    time = np.arange(data.shape[0])  # Assuming data has shape (number_sample, 3)
    time_in_seconds = time / sampling_freq  # Adjust sampling_rate as per your data

    plt.figure(figsize=(16, 8))
    for i in range(data.shape[1]):
        feature_name = data.columns[i]  # Extract feature name from DataFrame
        plt.plot(time_in_seconds, data.iloc[:, i], label=feature_name)
    
    plt.xlabel('Time (s)')
    plt.ylabel('IMU Measurements')
    plt.title('IMU Measurements over Time')
    plt.legend()
    plt.grid(True)
    plt.show()
    

def get_array_name(arr):
    """
    Returns the name of an array as a string.

    Args:
        arr: Array object.

    Returns:
        str: Name of the array.
    """
    frame = inspect.currentframe().f_back
    namespace = frame.f_globals.copy()
    namespace.update(frame.f_locals)
    
    for name, array in namespace.items():
        if id(array) == id(arr):
            return name
    
    return None


def resample_data_cubic_spline(raw_data, fs, fdesired):
    """
    Resamples the given data to the desired frequency using cubic spline interpolation.

    Parameters:
    raw_data (ndarray): The raw data to resample. Should have shape (num_samples, num_channels).
    fs (float): The sampling frequency of the raw data.
    fdesired (float): The desired resampling frequency.

    Returns:
    ndarray: The resampled data with shape (num_resampled_samples, num_channels).
    """

    raw_data = np.array(raw_data)

    # Reshape the input array if it has shape (n_samples,)
    if raw_data.ndim == 1:
        raw_data = raw_data.reshape(-1, 1)

    # Calculate the resampling factor
    resampling_factor = fs / fdesired

    # Define the time points for the original signal
    time_points = np.arange(raw_data.shape[0]) / fs

    # Define the time points for the resampled signal
    resampled_time_points = np.arange(0, time_points[-1], 1 / fdesired)

    # Initialize an empty array for the resampled data
    resampled_data = np.zeros((len(resampled_time_points), raw_data.shape[1]))

    # Loop over each column of the data and resample using cubic spline interpolation
    for i in range(raw_data.shape[1]):
        # Create a cubic spline interpolator object for this column
        interpolator = interpolate.interp1d(time_points, raw_data[:, i], kind='cubic')

        # Evaluate the interpolator at the resampled time points
        resampled_data[:, i] = interpolator(resampled_time_points)

    return resampled_data


def combine_dicts_to_dataframe(*dicts):
    """
    Combines multiple dictionaries into a single DataFrame.

    Args:
        *dicts: Variable number of dictionaries.

    Returns:
        df: DataFrame containing the combined data.
    """
    # Combine the dictionaries into a list
    data_list = list(dicts)

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(data_list)

    return df


def remove_extra_elements(array1, array2):
    size1 = array1.shape[0]
    size2 = array2.shape[0]

    if size1 > size2:
        trimmed_array1 = array1[:size2]
        return trimmed_array1, array2
    elif size2 > size1:
        trimmed_array2 = array2[:size1]
        return array1, trimmed_array2
    else:
        return array1, array2