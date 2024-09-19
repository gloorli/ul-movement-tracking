# Standard libraries
import os
import csv
import json
import inspect
from datetime import datetime
import re
import glob
import seaborn as sns
import matplotlib.font_manager as fm

# Data processing and analysis
import numpy as np
import pandas as pd
import subprocess
from collections import Counter

# Scientific computing
from scipy.signal import (butter, filtfilt, find_peaks, resample,
                          savgol_filter, freqz, lfilter)
from scipy.interpolate import CubicSpline, interp1d
from scipy.ndimage import zoom

# Visualization
import matplotlib.pyplot as plt

# Progress bar
from tqdm import tqdm


def create_folder(folder_path):
    """
    Creates a folder if it does not exist.
    :param folder_path: str, The path where the folder should be created.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def get_participant_info(participant_id, csv_path):
    """
    Extracts and returns a dictionary with data for the specified participant from a CSV file.

    Args:
    - participant_id (str): The ID of the participant whose data we want to extract.
    - csv_path (str): The path to the CSV file containing the participant data.

    Returns:
    - dict: A dictionary containing the data for the specified participant.
            Returns an empty dictionary if the participant ID is not found.
    """

    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Filter the DataFrame to get data for the specified participant
    participant_data = df[df['participant_id'] == participant_id]

    # If no data found, return an empty dictionary
    if participant_data.empty:
        return {}

    # Convert the filtered row to a dictionary
    # The `iloc[0]` ensures we're taking the first row (in case there are duplicates, which there shouldn't be)
    participant_dict = participant_data.iloc[0].to_dict()

    return participant_dict


def save_to_json(data, path):
    """
    Saves a dictionary to a JSON file named after the participant_id in the given path.

    Args:
    - data (dict): The data dictionary to save.
    - path (str): The directory where the file should be saved.

    Returns:
    None
    """
    def default_serialize(o):
        if isinstance(o, np.ndarray):
            return o.tolist()  # Convert ndarray to list
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

    participant_id = data['participant_id']
    filename = f"{participant_id}.json"
    full_path = os.path.join(path, filename)
    
    with open(full_path, 'w') as json_file:
        json.dump(data, json_file, indent=4, default=default_serialize)
    
    print(f"Data saved to: {full_path}")

    
def add_attributes_to_participant(participant_data, **attributes):
    """
    Adds multiple attributes with specified values to the participant's data dictionary.
    Converts lists to numpy arrays.

    Args:
    - participant_data (dict): The participant's data dictionary to be updated.
    - **attributes: Variable-length keyword arguments representing attribute names and their values.

    Returns:
    None
    """
    for attr_name, value in attributes.items():
        if isinstance(value, list):
            value = np.array(value)
        participant_data[attr_name] = value

        
def load_participant_json(participant_id, initial_path):
    """
    Loads the participant data from a JSON file.

    Args:
    - participant_id (str): The ID of the participant whose data needs to be loaded.
    - initial_path (str): The directory where the participant folders are located.

    Returns:
    - dict: The loaded participant data.
    """

    filename = f"{participant_id}.json"
    folder_path = os.path.join(initial_path, participant_id)
    full_path = os.path.join(folder_path, filename)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"No file found for participant ID: {participant_id} at {full_path}")

    with open(full_path, 'r') as json_file:
        data = json.load(json_file)

    return data


def get_attribute(participant_data, *fields):
    """
    Returns the desired attributes from the participant data.

    Args:
    - participant_data (dict): Participant data dictionary.
    - *fields: Fields to extract from the dictionary.

    Returns:
    - list: List of desired attributes. If an attribute is a tuple or list, it is converted to a numpy array.
    """
    results = []

    for field in fields:
        if field in participant_data:
            value = participant_data[field]
            if isinstance(value, (tuple, list)):
                value = np.array(value)
            results.append(value)
        else:
            results.append(None)

    return results


def get_dominant_hand(participant_data):
    """
    Returns the dominant hand for healthy participants 
    and the affected hand for stroke patients.

    Args:
    - participant_data (dict): Participant data dictionary.

    Returns:
    - str: 'dominant' or 'affected' hand.
    """
    if participant_data['participant_id'][0] == 'H':
        return participant_data['dominant_hand']
    elif participant_data['participant_id'][0] == 'S':
        # For stroke patients, we'll return the opposite of the affected hand.
        if participant_data['affected_hand'] == 'right':
            return 'left'
        elif participant_data['affected_hand'] == 'left':
            return 'right'
        else:
            raise ValueError("Invalid affected hand value.")
    else:
        raise ValueError("Invalid participant ID format.")


def get_json_paths(initial_path, group, excluded_participant=None):
    """
    Get all JSON paths for health or stroke based on the given group.

    Args:
    - initial_path (str): Initial directory to begin the search.
    - group (str): Either 'H' for health or 'S' for stroke.
    - excluded_participant (str, optional): The ID or name of the participant to exclude from the search.
  
    Returns:
    - list: List of paths for the desired JSON files.
    """
    
    if group not in ['H', 'S']:
        raise ValueError("Group must be either 'H' for health or 'S' for stroke.")

    # List directories starting with the given group
    directories = [d for d in os.listdir(initial_path) if os.path.isdir(os.path.join(initial_path, d)) and re.fullmatch(f"{group}\d+", d)]

    # If an excluded participant is specified, remove that directory from consideration
    if excluded_participant:
        directories = [d for d in directories if excluded_participant not in d]

    json_files = []

    for directory in directories:
        dir_path = os.path.join(initial_path, directory)
        # Search for .json files in the directory
        json_files.extend(glob.glob(os.path.join(dir_path, '*.json')))

    return json_files


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
        print(f"Array 1 has been trimmed to size {size2}.")
        return trimmed_array1, array2
    elif size2 > size1:
        trimmed_array2 = array2[:size1]
        print(f"Array 2 has been trimmed to size {size1}.")
        return array1, trimmed_array2
    else:
        print("Arrays already had same shape.")
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

def downsample_mask(ground_truth, original_rate=25, target_rate=1):
    """
    Downsample a 1D array from original_rate to target_rate by taking the most frequent value in each segment.

    Args:
        ground_truth (array-like): The original 1D array with ground truth data. Can contain numbers or strings.
        original_rate (int): The original sampling rate (default is 25Hz).
        target_rate (int): The target sampling rate (default is 1Hz).

    Returns:
        np.array: The downsampled 1D array, where each element is the most frequent value of the corresponding segment.
    """
    if original_rate % target_rate != 0:
        raise ValueError("Original frequency must be a multiple of the target frequency.")
    
    factor = original_rate // target_rate
    downsampled_array = []

    for i in range(0, len(ground_truth), factor):
        segment = ground_truth[i:i+factor]
        # Calculate the most frequent value
        most_frequent_value = Counter(segment).most_common(1)[0][0]
        downsampled_array.append(most_frequent_value)
    
    return np.array(downsampled_array)

def remove_excluded_frames(counts, pitch, task, GT_1Hz, exclusion_label=999):
    counts = counts[GT_1Hz != exclusion_label]
    pitch = pitch[GT_1Hz != exclusion_label]
    task = task[GT_1Hz != exclusion_label]
    GT_1Hz = GT_1Hz[GT_1Hz != exclusion_label]
    
    return counts, pitch, task, GT_1Hz

def remove_nan_frames(counts, pitch, task, GT_1Hz):
    nan_indices = np.isnan(counts)
    counts = counts[~nan_indices]
    pitch = pitch[~nan_indices]
    task = task[~nan_indices.ravel()]
    GT_1Hz = GT_1Hz[~nan_indices.ravel()]
    
    return counts, pitch, task, GT_1Hz

def extract_all_values_with_label(array, label_array, label_of_interest):
    extracted_array = array[label_array == label_of_interest]
            
    return extracted_array#, label_of_interest


def plot_resampled_arrays(original_mask, original_frequency, resampled_mask, desired_frequency):
    time_original = np.arange(len(original_mask)) / original_frequency
    time_resampled = np.arange(len(resampled_mask)) / desired_frequency

    plt.figure(figsize=(26, 6))

    plt.plot(time_original, original_mask, label=f'Original ({original_frequency}Hz)')
    plt.plot(time_resampled, resampled_mask, label=f'Resampled ({desired_frequency}Hz)', linestyle='--')
    plt.ylim(0, 1.1)
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
    
    # Validate inputs
    if ground_truth.ndim != 1 or predictions.ndim != 1:
        raise ValueError("Input arrays must be one-dimensional.")
    
    if not np.isin(ground_truth, [0, 1]).all() or not np.isin(predictions, [0, 1]).all():
        raise ValueError("Input arrays must contain only 0s and 1s.")
    
    # Convert inputs to NumPy arrays if they are not already
    ground_truth = np.array(ground_truth)
    predictions = np.array(predictions)

        # Ensure arrays have the same sizes
    ground_truth, predictions = remove_extra_elements(ground_truth, predictions)
    
    # Calculate evaluation metrics
    tp = np.sum((predictions == 1) & (ground_truth == 1))
    fp = np.sum((predictions == 1) & (ground_truth == 0))
    fn = np.sum((predictions == 0) & (ground_truth == 1))
    tn = np.sum((predictions == 0) & (ground_truth == 0))
    
    sensitivity = tp / (tp + fn) if tp + fn != 0 else 0
    specificity = tn / (tn + fp) if tn + fp != 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn != 0 else 0
    ppv = tp / (tp + fp) if tp + fp != 0 else 0
    npv = tn / (tn + fn) if tn + fn != 0 else 0
    
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

def get_mask_bilateral(GT_mask_ndh, GT_mask_dh):
    """
    Creates a bilateral mask by performing element-wise logical AND operation on the given NDH and DH masks.
    
    Args:
        GT_mask_ndh (list or ndarray): List or array representing the ground truth NDH mask.
        GT_mask_dh (list or ndarray): List or array representing the ground truth DH mask.
        
    Returns:
        ndarray: Bilateral mask where the value is 1 if and only if GT_mask_ndh AND GT_mask_dh row is 1; otherwise, it's 0.
    """
    
    # Convert input to numpy arrays if they are lists
    if isinstance(GT_mask_ndh, list):
        GT_mask_ndh = np.array(GT_mask_ndh)
    if isinstance(GT_mask_dh, list):
        GT_mask_dh = np.array(GT_mask_dh)
    
    # Check if the input arrays have the same shape
    assert GT_mask_ndh.shape == GT_mask_dh.shape, "The input arrays must have the same shape."
    
    # Perform element-wise logical AND operation on the masks
    mask_bilateral = np.logical_and(GT_mask_ndh, GT_mask_dh).astype(int)
    
    return mask_bilateral


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


def create_metrics_dictionary(metrics_ndh_CT, metrics_dh_CT,
                             metrics_ndh_OT=None, metrics_dh_OT=None, metrics_bilateral_CT=None, metrics_bilateral_OT=None):
    """
    Creates a dictionary with metrics data organized by the combination of vertical and horizontal axes.

    Args:
        (...)
    Returns:
        Dictionary with metrics data organized by the combination of vertical and horizontal axes.
    """

    data = {
        'OT_ndh_Sensitivity': metrics_ndh_OT['Sensitivity'],
        'OT_ndh_Specificity': metrics_ndh_OT['Specificity'],
        'OT_ndh_Accuracy': metrics_ndh_OT['Accuracy'],
        'OT_ndh_YoudenIndex': metrics_ndh_OT['Youden_Index'],

        'OT_dh_Sensitivity': metrics_dh_OT['Sensitivity'],
        'OT_dh_Specificity': metrics_dh_OT['Specificity'],
        'OT_dh_Accuracy': metrics_dh_OT['Accuracy'],
        'OT_dh_YoudenIndex': metrics_dh_OT['Youden_Index'],

        #'OT_bilateral_Sensitivity': metrics_bilateral_OT['Sensitivity'],
        #'OT_bilateral_Specificity': metrics_bilateral_OT['Specificity'],
        #'OT_bilateral_Accuracy': metrics_bilateral_OT['Accuracy'],
        #'OT_bilateral_PPV': metrics_bilateral_OT['PPV'],
        #'OT_bilateral_NPV': metrics_bilateral_OT['NPV'],

        'CT_ndh_Sensitivity': metrics_ndh_CT['Sensitivity'],
        'CT_ndh_Specificity': metrics_ndh_CT['Specificity'],
        'CT_ndh_Accuracy': metrics_ndh_CT['Accuracy'],
        'CT_ndh_YoudenIndex': metrics_ndh_CT['Youden_Index'],

        'CT_dh_Sensitivity': metrics_dh_CT['Sensitivity'],
        'CT_dh_Specificity': metrics_dh_CT['Specificity'],
        'CT_dh_Accuracy': metrics_dh_CT['Accuracy'],
        'CT_dh_YoudenIndex': metrics_dh_CT['Youden_Index'],

        #'CT_bilateral_Sensitivity': metrics_bilateral_CT['Sensitivity'],
        #'CT_bilateral_Specificity': metrics_bilateral_CT['Specificity'],
        #'CT_bilateral_Accuracy': metrics_bilateral_CT['Accuracy'],
        #'CT_bilateral_PPV': metrics_bilateral_CT['PPV'],
        #'CT_bilateral_NPV': metrics_bilateral_CT['NPV']
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


def plot_bar_chart(conventional_metrics, optimal_metrics, metric, scenario, group, save_filename=None, show_plot=True):
    
    sns.set(style="whitegrid")  # Use seaborn for a more modern look
    
    # Filter out keys that are in the excluded_keys set
    excluded_keys = {'PPV', 'NPV'}    
    metric_names = [key for key in conventional_metrics.keys() if key not in excluded_keys]
    num_metrics = len(metric_names)
    bar_width = 0.35
    ind = np.arange(num_metrics)  # X-axis locations for bars
    
    # Load a bold font for annotations
    prop = fm.FontProperties(weight='bold')

    fig, ax = plt.subplots(figsize=(10, 8))

    # Filtering the values based on the metric names
    conventional_values = [conventional_metrics[key] for key in metric_names]
    optimal_values = [optimal_metrics[key] for key in metric_names]
    
    # Calculate the maximum y-value needed
    max_y_value = max(max(conventional_values), max(optimal_values))
    ax.set_ylim(0, max_y_value * 1.2)  # Set y-limit to 120% of max y-value

    # Plot the bars
    rects1 = ax.bar(ind, conventional_values, bar_width, label='Conventional', color='royalblue')  # Modified color
    rects2 = ax.bar(ind + bar_width, optimal_values, bar_width, label='Optimal', color='forestgreen')  # Modified color
 
    if group == 'H':
        if scenario == 'ndh':
            scenario_plot = 'Non-Dom. H'
        elif scenario == 'dh':
            scenario_plot = 'Dom. H'
        else:
            scenario_plot = 'Bil'
    else:
        if scenario == 'ndh':
            scenario_plot = 'Aff. H'
        elif scenario == 'dh':
            scenario_plot = 'Non-Aff. H'
        else:
            scenario_plot = 'Bil'

    ax.set_xlabel('Metrics')
    ax.set_ylabel('Percentage')
    ax.set_title(f'Comparison of {metric} Metrics: Conventional vs Optimal [Side: {scenario_plot}]', fontsize=16)
    ax.set_xticks(ind + bar_width / 2)
    ax.set_xticklabels(metric_names)
    ax.legend(loc='upper right')  # Keep the legend inside

    # Annotate bars with rounded percentage values (as integers)
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{int(round(height))}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontproperties=prop)
            
    if save_filename:
        plt.savefig(save_filename)  # Save the plot to the specified file
        print('Figure saved')
    if show_plot:
        plt.tight_layout()
        plt.show()  # Show the plot if show_plot is True


def extract_fields_from_json_files(file_paths, fields):
    """
    Given a list of JSON file paths and a list of fields,
    return a dictionary containing NumPy arrays for each field
    collected from each JSON file.

    Parameters:
        - file_paths (list): List of paths to JSON files
        - fields (list): List of fields to be extracted from each JSON file

    Returns:
        - results (dict): Dictionary containing NumPy arrays of extracted values
                          for each field
    """
    
    # Initialize an empty list for each field to store values
    results = {field: [] for field in fields}

    for path in file_paths:
        # Validate file existence
        if not os.path.exists(path):
            print(f"Warning: File {path} does not exist.")
            continue

        # Load JSON file
        with open(path, 'r') as f:
            data = json.load(f)

        # Extract and store values of the specified fields
        for field in fields:
            value = data.get(field, None)
            if value is None:
                print(f"Warning: Field {field} does not exist in file {path}.")
            results[field].append(value)

    # Convert lists to NumPy arrays
    for field in fields:
        results[field] = np.array(results[field])

    return results


def get_data(folder, dominant_hand, using_axivity=False):
    # Load IMU data
    LW_data_filename = 'trimmed_LW_data.csv'
    chest_data_filename = 'trimmed_chest_data.csv'
    RW_data_filename = 'trimmed_RW_data.csv'

    LW_data_path = os.path.join(folder, LW_data_filename)
    chest_data_path = os.path.join(folder, chest_data_filename)
    RW_data_path = os.path.join(folder, RW_data_filename)

    LW_data = pd.read_csv(LW_data_path)
    chest_data = None
    if not using_axivity:
        chest_data = pd.read_csv(chest_data_path)
    RW_data = pd.read_csv(RW_data_path)

    # Load Video GT data
    GT_mask_LW_filename = 'GT_mask_LW.csv'
    GT_mask_RW_filename = 'GT_mask_RW.csv'

    GT_mask_LW_path = os.path.join(folder, GT_mask_LW_filename)
    GT_mask_RW_path = os.path.join(folder, GT_mask_RW_filename)

    GT_mask_LW = pd.read_csv(GT_mask_LW_path)
    GT_mask_RW = pd.read_csv(GT_mask_RW_path)

    # important: make the condition case-insensitive
    # for stroke: dh = non-affected hand 
    if dominant_hand.lower() == 'right': 
        dh_data = RW_data
        GT_mask_dh = np.array(GT_mask_RW)
        ndh_data = LW_data
        GT_mask_ndh = np.array(GT_mask_LW)
    else: 
        dh_data = LW_data
        GT_mask_dh = np.array(GT_mask_LW)
        ndh_data = RW_data
        GT_mask_ndh = np.array(GT_mask_RW)

    # Return the loaded dataframes
    return ndh_data, chest_data, dh_data, GT_mask_ndh, GT_mask_dh


def plot_superposition(array1, array2):
    """
    Plot two arrays over time using different colors with increased plot size.

    Parameters:
    array1 (ndarray): First array to be plotted.
    array2 (ndarray): Second array to be plotted.

    Raises:
    ValueError: If the sizes of the arrays are different.

    Returns:
    None
    """
    if len(array1) != len(array2):
        raise ValueError("Arrays must have the same size.")

    time = np.arange(len(array1))  # Time array assuming 1Hz sampling frequency

    plt.figure(figsize=(18, 9))  # Increased plot size

    plt.plot(time, array1, color='blue', label='Array 1')
    plt.plot(time, array2, color='red', label='Array 2')

    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Arrays Over Time')
    plt.legend()

    plt.show()


def replace_wbm_with_nf(mask):
    # Replace WBM mask by NF mask 
    mask = np.where(mask == -1, 0, mask)    
    return mask

def plot_angles_over_time(angles):
    """
    Plots angles over time.

    Args:
        angles (np.ndarray): Array of angles in degrees.
        sampling_freq (int): Sampling frequency in Hz.

    Returns:
        None.
    """
    
    sampling_freq = 50 #Hz 
    
    # Increase figure size
    plt.figure(figsize=(18, 9))
    
    # Calculate time array based on the length and sampling frequency
    time = np.arange(len(angles)) / sampling_freq

    # Plot angles over time
    plt.plot(time, angles)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Angle (degrees)')
    plt.title('Angles over Time')
    plt.grid(True)
    plt.show()

    
def from_LWRW_to_NDHDH(affected_hand, primitives):
    """
    Converts the list of LW/RW primitives to NDH and DH primitive list.

    Args:
        affected_hand (ndarray): The affected hand, either 'left' or 'right'.
        primitives (dict): A dictionary containing the primitive masks for both hands of all participants.

    Returns:
        tuple: A tuple containing two lists - `primitives_ndh` and `primitives_dh`.
            `primitives_ndh` (list): A list of primitive masks for the non-dominant hand.
            `primitives_dh` (list): A list of primitive masks for the dominant hand.

    Raises:
        ValueError: If the `affected_hand` value is neither 'right' nor 'left'.

    """
    primitives_ndh = []
    primitives_dh = []
    for i, affected_hand in enumerate(affected_hand):
        if affected_hand == 'left':
            primitives_ndh.append(primitives['primitive_mask_LW_25Hz'][i])
            primitives_dh.append(primitives['primitive_mask_RW_25Hz'][i])
        elif affected_hand == 'right':
            primitives_ndh.append(primitives['primitive_mask_RW_25Hz'][i])
            primitives_dh.append(primitives['primitive_mask_LW_25Hz'][i])
        else:
            raise ValueError("Invalid affected hand value. Must be 'right' or 'left'.")
    
    return primitives_ndh, primitives_dh

class ThesisStyle:
    """
    Class to define the visual style for the thesis.
    """
    def __init__(self):
        self.colours =  {
            'dark_blue': '#3D9BFF',
            'light_blue': '#9FCDFF',
            'yellow': '#FFCD00',
            'orange': '#F68400',
            'light_orange': '#ffa844',
            'affected': '#F68400',
            'pink': '#F87FB5',
            'turquoise': '#7CB7A4',
            'light_turquoise': '#9dc9bb',
            'healthy': '#7CB7A4',
            'grey': '#999999',
            'black_grey': '#404040',
            'black': '#000000',
            'white': '#FFFFFF'
        }
        self.label_colours = {
            'functional_movement': '#77DD77',
            'non_functional_movement': '#CA75E0',
            'reach': '#FFE599',
            'transport': '#CDA31E',
            'reposition': '#EE9C1B',
            'gesture': '#FFD966',
            'stabilization': '#F6A496',
            'idle': '#C85454',
            'arm_not_visible': self.colours['black_grey'],
            'open_bottle_and_pour_glass': self.colours['turquoise'],
            'drink': self.colours['turquoise'],
            'fold_rags_towels': self.colours['turquoise'],
            'sort_documents': self.colours['turquoise'],
            'brooming': self.colours['dark_blue'],
            'putting_on_and_off_coat': self.colours['dark_blue'],
            'keyboard_typing': self.colours['orange'],
            'stapling': self.colours['orange'],
            'walking': self.colours['black'], 
            'open_and_close_door': self.colours['pink'],
            'resting': self.colours['grey'],
            'other': self.colours['black'],
            'wipe_table': self.colours['turquoise'],
            'light_switch': self.colours['pink'],
        }

    def get_thesis_colours(self):
        """
        Returns a dictionary of colours used in the thesis.
        """
        return self.colours
    
    def get_label_colours(self):
        """
        Returns a dictionary of colours for different labels.
        """
        return self.label_colours
    
class CREATEid_to_thesisID:
    def __init__(self):
        self.conversion = {
            'S001': 'S01',
            'S008': 'S02',
            'S009': 'S03',
            'S012': 'S04',
            'S014': 'S05',
            'S015': 'S06',
            'S016': 'S07',
            'S018': 'S08',
            'S019': 'S09',
            'S020': 'S10',
            'S017': 'S11' #excluded
        }
    def get_thesisID(self, CREATEid):
        return self.conversion[CREATEid]
    
class task_labels:
    def __init__(self):
        self.task_labels = {
            'open_bottle_and_pour_glass': 'pour\nglas',
            'drink': 'drink',
            'fold_rags_towels': 'fold\nrags',
            'sort_documents': 'sort\ndocuments',
            'brooming': 'brooming',
            'putting_on_and_off_coat': 'coat',
            'keyboard_typing': 'keyboard\ntyping',
            'stapling': 'stapling',
            'walking': 'walking',
            'open_and_close_door': 'door',
            'resting': 'resting',
            'other': 'other',
            'wipe_table': 'wipe\ntable',
            'light_switch': 'light\nswitch'
        }
    def get_task_conversion(self):
        return self.task_labels
    def get_labelbos_task_labels(self):
        return list(self.task_labels.keys())
    def get_formated_task_labels(self):
        return list(self.task_labels.values())
    
thesis_style = ThesisStyle()
id_conversion = CREATEid_to_thesisID()
task_to_formated = task_labels()