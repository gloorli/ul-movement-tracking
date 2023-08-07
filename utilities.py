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
from scipy.ndimage import zoom
from scipy.interpolate import interp1d


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


def downsample_mask_interpolation(mask, original_fps, desired_fps):
    """
    Downsample a mask array from the original frames-per-second (fps) to the desired fps using interpolation.

    Parameters:
    mask (ndarray): The original mask array.
    original_fps (float): The original frames-per-second of the mask array.
    desired_fps (float): The desired frames-per-second for downsampling.

    Returns:
    ndarray: The downsampled mask array.
    """
    mask = np.array(mask)

    # Calculate the original and desired frame intervals
    original_interval = 1 / original_fps
    desired_interval = 1 / desired_fps

    # Create an array of original timestamps
    original_timestamps = np.arange(0, len(mask)) * original_interval

    # Create an array of desired timestamps
    desired_timestamps = np.arange(0, original_timestamps[-1], desired_interval)

    # Create an interpolation function based on the original timestamps and mask values
    mask_interpolation = interp1d(original_timestamps, mask.flatten(), kind='nearest', fill_value="extrapolate")

    # Use the interpolation function to obtain the downsampled mask values at desired timestamps
    downsampled_mask = mask_interpolation(desired_timestamps)

    # Round the interpolated values to the nearest integer (0 or 1)
    downsampled_mask = np.around(downsampled_mask).astype(int)

    return downsampled_mask


def resample_mask(mask, original_frequency, desired_frequency):
    if desired_frequency > original_frequency:
        # Upsample the mask using nearest neighbor interpolation
        zoom_factor = desired_frequency / original_frequency
        resampled_mask = zoom(mask, zoom_factor, order=0)
    else:
        # Downsample the mask using scipy.signal.resample
        num_data_points = int(len(mask) * desired_frequency / original_frequency)
        resampled_mask = np.round(resample(mask, num_data_points)).astype(int)
        
    # Ravel the resampled_mask before returning
    return np.ravel(resampled_mask)


def plot_resampled_arrays(original_mask, original_frequency, resampled_mask, desired_frequency):
    time_original = np.arange(len(original_mask)) / original_frequency
    time_resampled = np.arange(len(resampled_mask)) / desired_frequency

    plt.figure(figsize=(10, 6))

    plt.plot(time_original, original_mask, label=f'Original ({original_frequency}Hz)')
    plt.plot(time_resampled, resampled_mask, label=f'Resampled ({desired_frequency}Hz)', linestyle='--')

    plt.xlabel('Time (seconds)')
    plt.ylabel('Mask Value (0 or 1)')
    plt.title('Original and Resampled Mask')
    plt.legend()
    plt.grid(True)
    plt.show()


def save_optimal_threshold(file_path, ndh_threshold, dh_threshold, AC = True):
    # Create the file path
    if AC: 
        file_path = os.path.join(file_path, 'optimal_threshold_AC.csv')
    else: 
        file_path = os.path.join(file_path, 'optimal_threshold_GM.csv')

    try:
        # Open the file in write mode
        with open(file_path, 'w', newline='') as csvfile:
            # Create a CSV writer
            csv_writer = csv.writer(csvfile)

            # Write the header row with descriptions
            csv_writer.writerow(['Side', 'Threshold'])

            # Write the ndh threshold as a row
            csv_writer.writerow(['ndh', ndh_threshold])

            # Write the dh threshold as a row
            csv_writer.writerow(['dh', dh_threshold])

        print(f"Thresholds saved successfully at: {file_path}")
    except IOError as e:
        print(f"An error occurred while saving the thresholds: {e}")


def save_metrics_dictionary_as_csv(metrics_dictionary, folder, AC=True):
    """
    Saves the metrics dictionary as a CSV file in the specified folder.

    Args:
        metrics_dictionary: Dictionary with metrics data.
        folder: Folder path where the CSV file should be saved.
    """
    if AC:
        filename = os.path.join(folder, 'evaluation_metrics_AC.csv')
    else: 
        filename = os.path.join(folder, 'evaluation_metrics_GM.csv')
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write the data rows
        for (vertical, horizontal), value in metrics_dictionary.items():
            writer.writerow([vertical, horizontal, value])

    print(f"The metrics dictionary has been saved as {filename}.")


def read_csv_to_numpy(file_path1, file_path2):
    try:
        # Read CSV files into pandas DataFrames
        df1 = pd.read_csv(file_path1)
        df2 = pd.read_csv(file_path2)
    except FileNotFoundError:
        raise FileNotFoundError("One or both of the files does not exist.")

    # Convert DataFrames to NumPy arrays and flatten them using ravel()
    array1 = df1.to_numpy().ravel()
    array2 = df2.to_numpy().ravel()

    return array1, array2


def plot_radar_chart(conventional_metrics, optimal_metrics, metric, save_filename=None):
    metric_names = list(conventional_metrics.keys())
    num_metrics = len(metric_names)

    angles = [n / float(num_metrics) * 2 * np.pi for n in range(num_metrics)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

    # Extract the values from the metrics dictionaries
    conventional_values = [conventional_metrics[metric_name] for metric_name in metric_names]
    optimal_values = [optimal_metrics[metric_name] for metric_name in metric_names]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names)

    if metric == 'AC':
        # Plot the conventional metrics in blue
        conventional_values += conventional_values[:1]
        ax.plot(angles, conventional_values, 'o-', linewidth=2, label='Conventional Threshold', color='blue')
        ax.fill(angles, conventional_values, alpha=0.50, color='blue')

        # Plot the optimal metrics in green
        optimal_values += optimal_values[:1]
        ax.plot(angles, optimal_values, 'o-', linewidth=2, label='Optimal Threshold', color='green')
        ax.fill(angles, optimal_values, alpha=0.50, color='green')

        ax.set_title('Evaluation Metrics Comparison between Conventional vs Optimal AC Thresholds')
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

        # Annotate data points with percentage values for conventional metrics
        for angle, value, metric_name in zip(angles, conventional_values, metric_names):
            ax.annotate(f"{value:.2f}%", xy=(angle, value), xytext=(angle, value + 0.05),
                        horizontalalignment='center', verticalalignment='center')

        # Annotate data points with percentage values for optimal metrics
        for angle, value, metric_name in zip(angles, optimal_values, metric_names):
            ax.annotate(f"{value:.2f}%", xy=(angle, value), xytext=(angle, value + 0.05),
                        horizontalalignment='center', verticalalignment='center')
    if metric == 'GM':
        # Plot the conventional metrics in blue
        conventional_values += conventional_values[:1]
        ax.plot(angles, conventional_values, 'o-', linewidth=2, label='Conventional Functional Space', color='blue')
        ax.fill(angles, conventional_values, alpha=0.50, color='blue')

        # Plot the optimal metrics in green
        optimal_values += optimal_values[:1]
        ax.plot(angles, optimal_values, 'o-', linewidth=2, label='Optimal Functional Space', color='green')
        ax.fill(angles, optimal_values, alpha=0.50, color='green')

        ax.set_title('Evaluation Metrics Comparison between Conventional vs Optimal Functional Spaces for GM scores')
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

        # Annotate data points with percentage values for conventional metrics
        for angle, value, metric_name in zip(angles, conventional_values, metric_names):
            ax.annotate(f"{value:.2f}%", xy=(angle, value), xytext=(angle, value + 0.05),
                        horizontalalignment='center', verticalalignment='center')

        # Annotate data points with percentage values for optimal metrics
        for angle, value, metric_name in zip(angles, optimal_values, metric_names):
            ax.annotate(f"{value:.2f}%", xy=(angle, value), xytext=(angle, value + 0.05),
                        horizontalalignment='center', verticalalignment='center')
    if metric == 'GMAC':
        # Plot the conventional metrics in blue
        conventional_values += conventional_values[:1]
        ax.plot(angles, conventional_values, 'o-', linewidth=2, label='Conventional Functional Space', color='blue')
        ax.fill(angles, conventional_values, alpha=0.50, color='blue')

        # Plot the optimal metrics in green
        optimal_values += optimal_values[:1]
        ax.plot(angles, optimal_values, 'o-', linewidth=2, label='Optimal Functional Space', color='green')
        ax.fill(angles, optimal_values, alpha=0.50, color='green')

        ax.set_title('Evaluation Metrics Comparison between Conventional vs Optimal AC Threshold and FS for GMAC scores')
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

        # Annotate data points with percentage values for conventional metrics
        for angle, value, metric_name in zip(angles, conventional_values, metric_names):
            ax.annotate(f"{value:.2f}%", xy=(angle, value), xytext=(angle, value + 0.05),
                        horizontalalignment='center', verticalalignment='center')

        # Annotate data points with percentage values for optimal metrics
        for angle, value, metric_name in zip(angles, optimal_values, metric_names):
            ax.annotate(f"{value:.2f}%", xy=(angle, value), xytext=(angle, value + 0.05),
                        horizontalalignment='center', verticalalignment='center')
    plt.tight_layout()
    if save_filename:
        plt.savefig(save_filename)  # Save the plot to the specified file
    else:
        plt.show()  # Show the plot if save_filename is not provided


def resample_binary_mask(mask, original_frequency, desired_frequency):
    # Calculate the time steps of the original and desired frequencies
    original_time_step = 1 / original_frequency
    desired_time_step = 1 / desired_frequency

    # Calculate the time array for the original mask
    original_time_array = np.arange(0, len(mask)) * original_time_step

    # Calculate the time array for the resampled mask
    resampled_time_array = np.arange(0, len(mask) - 1, original_frequency / desired_frequency) * original_time_step

    # Create a CubicSpline object to perform interpolation
    cs = CubicSpline(original_time_array, mask)

    # Perform cubic spline interpolation on the resampled time array
    resampled_mask = cs(resampled_time_array)

    # Apply thresholding to make values binary (0 or 1)
    resampled_mask = np.where(resampled_mask >= 0.5, 1, 0)

    return np.array(resampled_mask, dtype=int)


def get_evaluation_metrics(ground_truth, predictions):
    """
    Calculates evaluation metrics for classification performance.

    Args:
        ground_truth: Numpy array of ground truth values (0s and 1s).
        predictions: Numpy array of predicted values (0s and 1s).

    Returns:
        A dictionary containing the evaluation metrics.
    """
    # Calculate evaluation metrics
    true_positives = np.sum(np.logical_and(predictions == 1, ground_truth == 1))
    false_positives = np.sum(np.logical_and(predictions == 1, ground_truth == 0))
    false_negatives = np.sum(np.logical_and(predictions == 0, ground_truth == 1))
    true_negatives = np.sum(np.logical_and(predictions == 0, ground_truth == 0))

    sensitivity = true_positives / (true_positives + false_negatives)
    specificity = true_negatives / (true_negatives + false_positives)
    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)

    # Calculate PPV (Positive Predictive Value) with denominator check
    positive_predictions = np.sum(predictions == 1)
    ppv = true_positives / positive_predictions if positive_predictions != 0 else 0

    # Calculate NPV (Negative Predictive Value) with denominator check
    negative_predictions = np.sum(predictions == 0)
    npv = true_negatives / negative_predictions if negative_predictions != 0 else 0

    # Calculate F1 Score
    f1_score = 2 * (ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) != 0 else 0

    # Calculate Youden Index
    youden_index = sensitivity + specificity - 1

    # Calculate False Positive Rate (FPR)
    fpr = false_positives / (false_positives + true_negatives)

    # Calculate False Negative Rate (FNR)
    fnr = false_negatives / (false_negatives + true_positives)

    # Convert metrics to percentages
    sensitivity *= 100
    specificity *= 100
    accuracy *= 100
    ppv *= 100
    npv *= 100
    f1_score *= 100
    fpr *= 100
    fnr *= 100
    youden_index *= 100

    return {
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Accuracy': accuracy,
        'PPV': ppv,
        'NPV': npv,
        'F1 Score': f1_score,
        'Youden Index': youden_index,
        'False Positive Rate': fpr,
        'False Negative Rate': fnr,
    }