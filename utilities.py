import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.signal import butter, filtfilt, find_peaks, resample, savgol_filter, freqz, lfilter
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
from scipy.interpolate import CubicSpline, interp1d
from scipy.stats import circmean
from scipy.ndimage import zoom
from ahrs.filters import Mahony, Madgwick
from ahrs.common import orientation
import pandas as pd
import csv
from datetime import datetime
from matplotlib.colors import ListedColormap
import os
import signal
import math
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


def save_optimal_threshold(file_path, ndh_threshold, dh_threshold, AC=True, group='H'):
    if AC:
        filename_prefix = 'optimal_threshold_AC'
    else:
        filename_prefix = 'optimal_threshold_GM'

    if group == 'H':
        filename_prefix = f'H_{filename_prefix}'
    elif group == 'S':
        filename_prefix = f'S_{filename_prefix}'
    else:
        raise ValueError("Invalid group parameter. Use 'H' for healthy or 'S' for stroke.")

    file_name = f"{filename_prefix}.csv"
    file_path = os.path.join(file_path, file_name)

    try:
        with open(file_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Side', 'Threshold'])
            csv_writer.writerow(['ndh', ndh_threshold])
            csv_writer.writerow(['dh', dh_threshold])

        print(f"Thresholds saved successfully at: {file_path}")
    except IOError as e:
        print(f"An error occurred while saving the thresholds: {e}")


def save_metrics_dictionary_as_csv(metrics_dictionary, folder, metric):
    """
    Saves the metrics dictionary as a CSV file in the specified folder.

    Args:
        metrics_dictionary: Dictionary with metrics data.
        folder: Folder path where the CSV file should be saved.
        metric: The type of metric ('AC', 'GM', 'GMAC').
    Raises:
        ValueError: If the metric provided is not one of 'AC', 'GM', or 'GMAC'.
    """
    if metric == 'AC':
        filename = os.path.join(folder, 'evaluation_metrics_AC.csv')
    elif metric == 'GM': 
        filename = os.path.join(folder, 'evaluation_metrics_GM.csv')
    elif metric == 'GMAC': 
        filename = os.path.join(folder, 'evaluation_metrics_GMAC.csv')
    else:
        raise ValueError("Invalid metric. Metric should be one of 'AC', 'GM', or 'GMAC'.")

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

    # Ensure arrays have same sizes 
    ground_truth, predictions = remove_extra_elements(ground_truth, predictions)

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

    # Convert metrics to percentages
    sensitivity *= 100
    specificity *= 100
    accuracy *= 100
    ppv *= 100
    npv *= 100

    return {
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Accuracy': accuracy,
        'PPV': ppv,
        'NPV': npv,
    }


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


def create_metrics_dictionary(metrics_ndh_CT, metrics_dh_CT, metrics_bilateral_CT,
                             metrics_ndh_OT, metrics_dh_OT, metrics_bilateral_OT):
    """
    Creates a dictionary with metrics data organized by the combination of vertical and horizontal axes.

    Args:
        metrics_ndh_CT: Metrics data for ndh and CT.
        metrics_dh_CT: Metrics data for dh and CT.
        metrics_bilateral_CT: Metrics data for bilateral and CT.
        metrics_ndh_OT: Metrics data for ndh and OT.
        metrics_dh_OT: Metrics data for dh and OT.
        metrics_bilateral_OT: Metrics data for bilateral and OT.

    Returns:
        Dictionary with metrics data organized by the combination of vertical and horizontal axes.
    """
    data = {
        ('OT_ndh', 'Sensitivity'): metrics_ndh_OT['Sensitivity'],
        ('OT_ndh', 'Specificity'): metrics_ndh_OT['Specificity'],
        ('OT_ndh', 'Accuracy'): metrics_ndh_OT['Accuracy'],
        ('OT_ndh', 'PPV'): metrics_ndh_OT['PPV'],
        ('OT_ndh', 'NPV'): metrics_ndh_OT['NPV'],
        ('OT_dh', 'Sensitivity'): metrics_dh_OT['Sensitivity'],
        ('OT_dh', 'Specificity'): metrics_dh_OT['Specificity'],
        ('OT_dh', 'Accuracy'): metrics_dh_OT['Accuracy'],
        ('OT_dh', 'PPV'): metrics_dh_OT['PPV'],
        ('OT_dh', 'NPV'): metrics_dh_OT['NPV'],
        ('OT_bilateral', 'Sensitivity'): metrics_bilateral_OT['Sensitivity'],
        ('OT_bilateral', 'Specificity'): metrics_bilateral_OT['Specificity'],
        ('OT_bilateral', 'Accuracy'): metrics_bilateral_OT['Accuracy'],
        ('OT_bilateral', 'PPV'): metrics_bilateral_OT['PPV'],
        ('OT_bilateral', 'NPV'): metrics_bilateral_OT['NPV'],
        ('CT_ndh', 'Sensitivity'): metrics_ndh_CT['Sensitivity'],
        ('CT_ndh', 'Specificity'): metrics_ndh_CT['Specificity'],
        ('CT_ndh', 'Accuracy'): metrics_ndh_CT['Accuracy'],
        ('CT_ndh', 'PPV'): metrics_ndh_CT['PPV'],
        ('CT_ndh', 'NPV'): metrics_ndh_CT['NPV'],
        ('CT_dh', 'Sensitivity'): metrics_dh_CT['Sensitivity'],
        ('CT_dh', 'Specificity'): metrics_dh_CT['Specificity'],
        ('CT_dh', 'Accuracy'): metrics_dh_CT['Accuracy'],
        ('CT_dh', 'PPV'): metrics_dh_CT['PPV'],
        ('CT_dh', 'NPV'): metrics_dh_CT['NPV'],
        ('CT_bilateral', 'Sensitivity'): metrics_bilateral_CT['Sensitivity'],
        ('CT_bilateral', 'Specificity'): metrics_bilateral_CT['Specificity'],
        ('CT_bilateral', 'Accuracy'): metrics_bilateral_CT['Accuracy'],
        ('CT_bilateral', 'PPV'): metrics_bilateral_CT['PPV'],
        ('CT_bilateral', 'NPV'): metrics_bilateral_CT['NPV']
    }

    return data


def extract_age(csv_file_path):
    """
    Extracts and returns the ages of participants from a CSV file, categorized by 'H' and 'S'.

    Args:
        csv_file_path (str): Path to the CSV file containing participant IDs and ages.

    Returns:
        A dictionary with 'H' and 'S' as keys, and corresponding NumPy arrays of ages as values.
    """
    age_data = {'H': [], 'S': []}

    # Load CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    for index, row in df.iterrows():
        participant_id = row['participant_id']
        age = row['age']

        if participant_id.startswith('H'):
            age_data['H'].append(age)
        elif participant_id.startswith('S'):
            age_data['S'].append(age)

    # Convert lists to NumPy arrays
    age_data['H'] = np.array(age_data['H'])
    age_data['S'] = np.array(age_data['S'])

    return age_data


def side_by_side_box_plot(data1, data2, labels=None, x_axis_labels=None):
    """
    Plots two arrays as side-by-side vertical box plots on the same plot.

    Args:
        data1 (numpy.ndarray): First array of data.
        data2 (numpy.ndarray): Second array of data.
        labels (list, optional): Labels for the two box plots.
        x_axis_labels (list, optional): Labels for the x-axis.

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed

    box_plot_data = [data1, data2]
    box_plot = plt.boxplot(box_plot_data, vert=True, patch_artist=True, labels=labels)

    if x_axis_labels:
        plt.xticks(range(1, len(x_axis_labels) + 1), x_axis_labels)

    plt.ylabel("Age")
    plt.title("Age Distribution Comparison between Healthy and Stroke groups")
    plt.grid(True)
    plt.show()


def load_data_from_csv(folder, data_type):
    """
    Load AC values, mask values, or GM arrays from CSV files.

    Args:
        folder (str): Folder path containing the CSV files.
        data_type (str): Flag to indicate whether to load 'AC', 'mask', or 'GM' data.

    Returns:
        Data loaded from CSV files.
    """
    if data_type == 'AC':
        ndh_output_filename = 'count_brond_ndh.csv'
        dh_output_filename = 'count_brond_dh.csv'
        column_name = 'AC Brond'
    elif data_type == 'mask':
        ndh_output_filename = 'GT_mask_ndh_1Hz.csv'
        dh_output_filename = 'GT_mask_dh_1Hz.csv'
        column_name = 'mask'
    elif data_type == 'GM':
        gm_output_filename = 'gm_datasets.csv'
        gm_output_path = os.path.join(folder, gm_output_filename)
        if not os.path.exists(gm_output_path):
            raise ValueError("GM CSV file not found in the specified folder.")
        gm_df = pd.read_csv(gm_output_path)
        pitch_mad_ndh = gm_df['Pitch MAD NDH'].to_numpy()
        yaw_mad_ndh = gm_df['Yaw MAD NDH'].to_numpy()
        pitch_mad_dh = gm_df['Pitch MAD DH'].to_numpy()
        yaw_mad_dh = gm_df['Yaw MAD DH'].to_numpy()
        GT_mask_50Hz_ndh = gm_df['GT Mask 50Hz NDH'].to_numpy()
        GT_mask_50Hz_dh = gm_df['GT Mask 50Hz DH'].to_numpy()
        return pitch_mad_ndh, yaw_mad_ndh, pitch_mad_dh, yaw_mad_dh, GT_mask_50Hz_ndh, GT_mask_50Hz_dh
    else:
        raise ValueError("Invalid data_type. Should be one of 'AC', 'mask', or 'GM'.")

    ndh_output_path = os.path.join(folder, ndh_output_filename)
    dh_output_path = os.path.join(folder, dh_output_filename)

    if not os.path.exists(ndh_output_path) or not os.path.exists(dh_output_path):
        raise ValueError(f"{data_type} CSV files not found in the specified folder.")

    if data_type == 'AC':
        count_brond_ndh = pd.read_csv(ndh_output_path)[column_name].to_numpy()
        count_brond_dh = pd.read_csv(dh_output_path)[column_name].to_numpy()
        return count_brond_ndh, count_brond_dh
    elif data_type == 'mask':
        GT_mask_ndh_1Hz = pd.read_csv(ndh_output_path)[column_name]
        GT_mask_dh_1Hz = pd.read_csv(dh_output_path)[column_name]
        return np.array(GT_mask_ndh_1Hz), np.array(GT_mask_dh_1Hz)

    
def load_testing_data(testing_participant_path):
    """
    Load various data arrays for testing.

    Args:
        testing_participant_path (str): Path to the participant's data folder.

    Returns:
        Tuple of NumPy arrays containing testing data.
    """
    # Load AC and GT @ 1 Hz
    testing_count_brond_ndh, testing_count_brond_dh = load_data_from_csv(testing_participant_path, data_type='AC')
    testing_GT_mask_ndh_1Hz, testing_GT_mask_dh_1Hz = load_data_from_csv(testing_participant_path, data_type='mask')
    testing_GT_mask_bil_1Hz = get_mask_bilateral(testing_GT_mask_ndh_1Hz, testing_GT_mask_dh_1Hz)

    # Load angles and GT @ 50 Hz
    testing_pitch_mad_ndh, testing_yaw_mad_ndh, testing_pitch_mad_dh, testing_yaw_mad_dh, testing_GT_mask_50Hz_ndh, testing_GT_mask_50Hz_dh = load_data_from_csv(testing_participant_path, data_type='GM')
    testing_GT_mask_bil_50Hz = get_mask_bilateral(testing_GT_mask_50Hz_ndh, testing_GT_mask_50Hz_dh)

    return (testing_count_brond_ndh, testing_count_brond_dh,
            testing_GT_mask_ndh_1Hz, testing_GT_mask_dh_1Hz, testing_GT_mask_bil_1Hz,
            testing_pitch_mad_ndh, testing_yaw_mad_ndh, testing_pitch_mad_dh, testing_yaw_mad_dh,
            testing_GT_mask_50Hz_ndh, testing_GT_mask_50Hz_dh, testing_GT_mask_bil_50Hz)


def find_specific_csv_files(initial_path, csv_file_names, participant_group, testing_participant=None):
    """
    Searches for specific CSV files inside folders starting with 'H' or 'S' within the 'CreateStudy' directory and its subdirectories.

    Args:
        initial_path (str): The path to the 'CreateStudy' directory.
        csv_file_names (list): A list of CSV file names to search for.
        participant_group (str): The participant group to search for ('H' for healthy, 'S' for stroke).
        testing_participant (str, optional): The participant ID to exclude from the analysis.

    Returns:
        A dictionary containing lists of paths to CSV files for each requested file name.
    """
    csv_files_dict = {csv_name: [] for csv_name in csv_file_names}

    # Walk through the directory tree starting from initial_path
    for root, dirs, files in os.walk(initial_path):
        for dir_name in dirs:
            # Check if the folder starts with participant_group
            if dir_name.startswith(participant_group):
                if testing_participant is None or dir_name != testing_participant:  # Exclude the testing participant if provided
                    folder_path = os.path.join(root, dir_name)
                    # Find the specific CSV files inside each folder
                    for csv_name in csv_file_names:
                        csv_file_path = os.path.join(folder_path, csv_name)
                        if os.path.isfile(csv_file_path):
                            csv_files_dict[csv_name].append(csv_file_path)

    return csv_files_dict


def get_mask_bilateral(GT_mask_ndh, GT_mask_dh):
    """
    Creates a bilateral mask by performing element-wise logical AND operation on the given left and dh masks.
    
    Args:
        GT_mask_ndh (ndarray): Array representing the ground truth ndh mask.
        GT_mask_dh (ndarray): Array representing the ground truth dh mask.
        
    Returns:
        ndarray: Bilateral mask where the value is 1 if and only if GT_mask_ndh AND GT_mask_dh row is 1; othedhise, it's 0.
    """
    # Check if the input arrays have the same shape
    assert GT_mask_ndh.shape == GT_mask_dh.shape, "The input arrays must have the same shape."
    
    # Perform element-wise logical AND operation on the masks
    mask_bilateral = np.logical_and(GT_mask_ndh, GT_mask_dh).astype(int)
    
    return mask_bilateral


def extract_data_screening_participant(csv_file_path):
    """
    Extracts and returns participant data from a CSV file, categorized by 'H' (healthy) and 'S' (stroke).

    Args:
        csv_file_path (str): Path to the CSV file containing participant data.

    Returns:
        A dictionary with 'H' and 'S' as keys, each containing nested dictionaries with field names as keys
        and corresponding NumPy arrays of data as values.
    """
    fields = ['participant_id', 'age', 'dominant_hand', 'affected_hand', 'ARAT_score']
    data = {'H': {field: [] for field in fields}, 'S': {field: [] for field in fields}}

    # Load CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    for index, row in df.iterrows():
        participant_id = row['participant_id']
        category = participant_id[0]
        
        for field in fields:
            data[category][field].append(row[field])

    # Convert lists to NumPy arrays
    for category in ['H', 'S']:
        for field in fields:
            data[category][field] = np.array(data[category][field])

    return data


def plot_correlation(x_array, threshold_values, x_value='X'):
    # Calculate Spearman correlation coefficient and p-value
    correlation_coef, p_value = spearmanr(x_array, threshold_values)

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x_array, threshold_values, color='blue', label='Data Points')
    
    # Fit a linear regression line
    fit = np.polyfit(x_array, threshold_values, deg=1)
    plt.plot(x_array, np.polyval(fit, x_array), color='red', label='Linear Fit')
    
    plt.title('Correlation Plot')
    plt.xlabel(x_value)
    plt.ylabel('Threshold Values')
    plt.legend()
    plt.grid(True)
    
    # Display correlation coefficient and significance level
    plt.text(0.05, 0.9, f'Spearman Correlation Coefficient: {correlation_coef:.2f}\nSignificance Level (p-value): {p_value:.3f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    
    # Determine and print correlation strength
    if p_value < 0.05:
        if correlation_coef > 0.7 or correlation_coef < -0.7:
            correlation_strength = "Strong"
        elif abs(correlation_coef) > 0.3:
            correlation_strength = "Moderate"
        else:
            correlation_strength = "Weak"
        print(f"Correlation Strength: {correlation_strength}")
    else:
        print("Correlation is not statistically significant (p >= 0.05)")
    
    plt.show()