import os
import csv
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
from activity_count_function import *
from utilities import *
from individual_analysis_ac_functions import *
from individual_analysis_fs_functions import *


def regroup_field_data_metrics(csv_files):
    """
    Regroups the data from multiple participants into arrays per field cross participants.

    Args:
        csv_files: List of file paths to the CSV files for each participant.

    Returns:
        Dictionary of arrays per field cross participants.
    """
    field_data = {}

    for csv_file in csv_files:
        with open(csv_file, 'r') as csvfile:
            reader = csv.reader(csvfile)

            for row in reader:
                vertical, horizontal, value = row
                field_key = f"{vertical}_{horizontal}"
                
                if field_key not in field_data:
                    field_data[field_key] = []

                field_data[field_key].append(float(value))

    return field_data


def extract_data_from_csv(paths):
    ndh_values = []
    dh_values = []

    for path in paths:
        with open(path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            header = reader.fieldnames
            if 'Side' not in header:
                print(f"Error: 'Side' column not found in {path}")
                continue
            side_column = 'Side'

            if 'Threshold' in header:
                threshold_column = 'Threshold'
            elif 'Functional Space' in header:
                threshold_column = 'Functional Space'
            else:
                print(f"Error: 'Threshold' or 'Functional Space' column not found in {path}")
                continue

            for row in reader:
                side = row[side_column].strip().lower()
                if side == 'ndh':
                    ndh_values.append(float(row[threshold_column]))
                elif side == 'dh':
                    dh_values.append(float(row[threshold_column]))

    return ndh_values, dh_values


def plot_side_metrics(data_dict, metric_names):
    for metric_name in metric_names:
        ndh_data_ot = []
        ndh_data_ct = []
        dh_data_ot = []
        dh_data_ct = []
        bilateral_data_ot = []
        bilateral_data_ct = []

        for key, value in data_dict.items():
            parts = key.split('_')
            if len(parts) == 3 and parts[2] == metric_name:
                if parts[0] == 'OT' and parts[1] == 'ndh':
                    ndh_data_ot.extend(value)
                elif parts[0] == 'CT' and parts[1] == 'ndh':
                    ndh_data_ct.extend(value)
                elif parts[0] == 'OT' and parts[1] == 'dh':
                    dh_data_ot.extend(value)
                elif parts[0] == 'CT' and parts[1] == 'dh':
                    dh_data_ct.extend(value)
                elif parts[0] == 'OT' and parts[1] == 'bilateral':
                    bilateral_data_ot.extend(value)
                elif parts[0] == 'CT' and parts[1] == 'bilateral':
                    bilateral_data_ct.extend(value)

        if not ndh_data_ot or not ndh_data_ct or not dh_data_ot or not dh_data_ct or not bilateral_data_ot or not bilateral_data_ct:
            print(f"Data not found for the metric: {metric_name}")
            continue
        # Plotting
        plot_data_side_by_side(ndh_data_ct, ndh_data_ot, dh_data_ct, dh_data_ot, bilateral_data_ct, bilateral_data_ot, metric_name)


def plot_data_side_by_side(data1_ct, data1_ot, data2_ct, data2_ot, data3_ct, data3_ot, metric_name):
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))

    positions = [1, 2, 4, 5, 7, 8]
    width = 0.2

    # Plot 'ndh' side
    plt.boxplot([data1_ct, data1_ot], positions=positions[:2], labels=['NDH CT', 'NDH OT'], patch_artist=True, widths=width, whis=2)
    # Plot 'dh' side
    plt.boxplot([data2_ct, data2_ot], positions=positions[2:4], labels=['DH CT', 'DH OT'], patch_artist=True, widths=width, whis=2)
    # Plot 'bilateral' side
    plt.boxplot([data3_ct, data3_ot], positions=positions[4:], labels=['Bilateral CT', 'Bilateral OT'], patch_artist=True, widths=width, whis=2)

    plt.title(f'Distribution of {metric_name} across individuals for ndh, dh, and Bilateral UL usage (CT vs OT)')
    plt.xlabel('Side')
    plt.ylabel(metric_name)

    colors = ['lightblue', 'lightgreen', 'lightblue', 'lightgreen', 'lightblue', 'lightgreen']
    for patch, color in zip(plt.gca().patches, colors):
        patch.set_facecolor(color)

    # Create the legend
    legend_elements = [plt.Rectangle((0, 0), 1, 1, color='lightblue', label='Conventional Threshold'),
                       plt.Rectangle((0, 0), 1, 1, color='lightgreen', label='Optimal Threshold')]
    plt.legend(handles=legend_elements, loc='upper left')

    plt.tight_layout()
    plt.show()


def resample_AC(AC, original_frequency, desired_frequency):
    # Calculate the time steps of the original and desired frequencies
    original_time_step = 1 / original_frequency
    desired_time_step = 1 / desired_frequency

    # Calculate the time array for the original AC
    original_time_array = np.arange(0, len(AC)) * original_time_step

    # Calculate the time array for the resampled AC
    resampled_time_array = np.arange(0, len(AC) - 1, original_frequency / desired_frequency) * original_time_step

    # Create a CubicSpline object to perform interpolation
    cs = CubicSpline(original_time_array, AC)

    # Perform cubic spline interpolation on the resampled time array
    resampled_AC = cs(resampled_time_array)

    # Set negative values to zero since AC are positive only 
    resampled_AC[resampled_AC < 0] = 0

    return np.array(resampled_AC)


def get_group_dataset_from_csv(csv_files, mask=True):
    """
    Resamples data from CSV files to match the smallest length among the arrays.

    Args:
        csv_files (list): List of CSV file paths.
        mask (bool): Boolean flag indicating whether the data is a mask.

    Returns:
        np.ndarray: NumPy array of resampled data arrays.
    """
    all_data = []
    min_length = float('inf')
    elements_removed = []  # List to store the number of elements removed for each array
    resampled_data = []

    for csv_file in csv_files:
        # Read the CSV file and extract the data
        df = pd.read_csv(csv_file)
        data = df.iloc[:, 0].values
        # Update the minimum length
        min_length = min(min_length, len(data))
        all_data.append(data)
        print(data.shape)

    for data in all_data:
        
        # First get the original_frequency and desired_frequency given by the minimum array size to achieve
        original_frequency = len(data)
        desired_frequency = min_length
        
        if mask: 
            # Downsample the binary mask 
            # Then downsample the mask 
            if desired_frequency != original_frequency:
                resampled_values = resample_binary_mask(data, original_frequency, desired_frequency)
            else: 
                resampled_values = data
            
            # Plot the data before and after resampling 
            plot_resampled_arrays(data, original_frequency, resampled_values, desired_frequency)

            # Save the new resampled values inside the resampled_data array 
            resampled_data.append(resampled_values)

            # Print the number of elements removed by this resampling operation 
            num_removed = len(data) - len(resampled_values)
            elements_removed.append(num_removed)

        else: 
            
            # Case of AC resampling
            if desired_frequency != original_frequency:
                resampled_values = resample_AC(data, original_frequency, desired_frequency)
            else: 
                resampled_values = data
                
            # Plot the data before and after resampling 
            plot_resampled_arrays(data, original_frequency, resampled_values, desired_frequency)
            
            # Save the new resampled values inside the resampled_data array 
            resampled_data.append(resampled_values)
            
            # Print the number of elements removed by this resampling operation 
            num_removed = len(data) - len(resampled_values)
            elements_removed.append(num_removed)

    group_data = np.concatenate(resampled_data, axis=0)
    
    # Print the number of elements removed for each array
    for idx, num_removed in enumerate(elements_removed, start=1):
        print(f"Elements removed in array {idx}: {num_removed}")

    return group_data


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


def compute_evaluation_metrics_ac(testing_count_brond_ndh, testing_count_brond_dh,
                                  testing_GT_mask_ndh_1Hz, testing_GT_mask_dh_1Hz, testing_GT_mask_bil_1Hz,
                                  conventional_threshold_unilateral, opt_threshold_ndh, opt_threshold_dh):
    """
    Compute evaluation metrics for different prediction scenarios.

    Args:
        testing_count_brond_ndh (numpy.ndarray): NumPy array containing AC values for the left bronchus (NDH).
        testing_count_brond_dh (numpy.ndarray): NumPy array containing AC values for the right bronchus (DH).
        testing_GT_mask_ndh_1Hz (pandas.Series): Series containing GT mask values for the left bronchus (NDH).
        testing_GT_mask_dh_1Hz (pandas.Series): Series containing GT mask values for the right bronchus (DH).
        testing_GT_mask_bil_1Hz (pandas.Series): Series containing bilateral GT mask values.
        conventional_threshold_unilateral (float): Conventional AC threshold for unilateral prediction.
        opt_threshold_ndh (float): Optimal AC threshold for NDH prediction.
        opt_threshold_dh (float): Optimal AC threshold for DH prediction.
        metric_name (str): Name of the metric being compared ('AC', 'GM', or 'GMAC').

    Returns:
        A tuple containing two dictionaries:
        - AC scores dictionary
        - Evaluation metrics dictionary
    """
    # Compute predictions
    testing_ac_ndh_conv = get_prediction_ac(testing_count_brond_ndh, conventional_threshold_unilateral)
    testing_ac_ndh_opt = get_prediction_ac(testing_count_brond_ndh, opt_threshold_ndh)
    testing_ac_dh_conv = get_prediction_ac(testing_count_brond_dh, conventional_threshold_unilateral)
    testing_ac_dh_opt = get_prediction_ac(testing_count_brond_dh, opt_threshold_dh)
    testing_ac_bil_conv = get_prediction_bilateral(testing_count_brond_ndh, conventional_threshold_unilateral,
                                                   testing_count_brond_dh, conventional_threshold_unilateral)
    testing_ac_bil_opt = get_prediction_bilateral(testing_count_brond_ndh, opt_threshold_ndh,
                                                  testing_count_brond_dh, opt_threshold_dh)

    # Compute AC score dictionaries
    ac_scores = {
        'ndh_conv': testing_ac_ndh_conv,
        'ndh_opt': testing_ac_ndh_opt,
        'dh_conv': testing_ac_dh_conv,
        'dh_opt': testing_ac_dh_opt,
        'bil_conv': testing_ac_bil_conv,
        'bil_opt': testing_ac_bil_opt
    }

    # Compute evaluation metrics
    eval_metrics_ndh_conv = get_evaluation_metrics(testing_GT_mask_ndh_1Hz, testing_ac_ndh_conv)
    eval_metrics_ndh_opt = get_evaluation_metrics(testing_GT_mask_ndh_1Hz, testing_ac_ndh_opt)
    eval_metrics_dh_conv = get_evaluation_metrics(testing_GT_mask_dh_1Hz, testing_ac_dh_conv)
    eval_metrics_dh_opt = get_evaluation_metrics(testing_GT_mask_dh_1Hz, testing_ac_dh_opt)
    eval_metrics_bil_conv = get_evaluation_metrics(testing_GT_mask_bil_1Hz, testing_ac_bil_conv)
    eval_metrics_bil_opt = get_evaluation_metrics(testing_GT_mask_bil_1Hz, testing_ac_bil_opt)

    # Create evaluation metrics dictionary
    eval_metrics = {
        'ndh_conv': eval_metrics_ndh_conv,
        'ndh_opt': eval_metrics_ndh_opt,
        'dh_conv': eval_metrics_dh_conv,
        'dh_opt': eval_metrics_dh_opt,
        'bil_conv': eval_metrics_bil_conv,
        'bil_opt': eval_metrics_bil_opt
    }

    return ac_scores, eval_metrics


def compute_evaluation_metrics_gm(testing_pitch_mad_ndh, testing_pitch_mad_dh,
                                  testing_yaw_mad_ndh, testing_yaw_mad_dh,
                                  testing_GT_mask_2Hz_ndh, testing_GT_mask_2Hz_dh, testing_GT_mask_2Hz_bil,
                                  conventional_functional_space, group_optimal_fs_ndh, group_optimal_fs_dh,):
    """
    Compute evaluation metrics for different prediction scenarios using the GM algorithm.

    Args:
        testing_pitch_mad_ndh (numpy.ndarray): NumPy array containing pitch_mad values for the left bronchus (NDH).
        testing_pitch_mad_dh (numpy.ndarray): NumPy array containing pitch_mad values for the right bronchus (DH).
        testing_yaw_mad_ndh (numpy.ndarray): NumPy array containing yaw_mad values for the left bronchus (NDH).
        testing_yaw_mad_dh (numpy.ndarray): NumPy array containing yaw_mad values for the right bronchus (DH).
        testing_GT_mask_2Hz_ndh (pandas.Series): Series containing GT mask values for the left bronchus (NDH).
        testing_GT_mask_2Hz_dh (pandas.Series): Series containing GT mask values for the right bronchus (DH).
        testing_GT_mask_2Hz_bil (pandas.Series): Series containing bilateral GT mask values.
        conventional_functional_space (int): Conventional functional space parameter.
        group_optimal_fs_ndh (function): Group optimal functional space function for NDH prediction.
        group_optimal_fs_dh (function): Group optimal functional space function for DH prediction.
        metric_name (str): Name of the metric being compared ('AC', 'GM', or 'GMAC').

    Returns:
        A tuple containing two dictionaries:
        - GM scores dictionary
        - Evaluation metrics dictionary
    """
    # Compute predictions @ 2Hz using GM algorithm
    testing_gm_ndh_conv = gm_algorithm(testing_pitch_mad_ndh, testing_yaw_mad_ndh, conventional_functional_space)
    testing_gm_ndh_opt = gm_algorithm(testing_pitch_mad_ndh, testing_yaw_mad_ndh, group_optimal_fs_ndh)
    testing_gm_dh_conv = gm_algorithm(testing_pitch_mad_dh, testing_yaw_mad_dh, conventional_functional_space)
    testing_gm_dh_opt = gm_algorithm(testing_pitch_mad_dh, testing_yaw_mad_dh, group_optimal_fs_dh)
    testing_gm_bil_conv = get_mask_bilateral(testing_gm_ndh_conv, testing_gm_dh_conv)
    testing_gm_bil_opt = get_mask_bilateral(testing_gm_ndh_opt, testing_gm_dh_opt)
    
    # Compute GM score dictionaries
    gm_scores = {
        'ndh_conv': testing_gm_ndh_conv,
        'ndh_opt': testing_gm_ndh_opt,
        'dh_conv': testing_gm_dh_conv,
        'dh_opt': testing_gm_dh_opt,
        'bil_conv': testing_gm_bil_conv,
        'bil_opt': testing_gm_bil_opt
    }

    # Compute evaluation metrics by comparing masks and GM scores @ 2 Hz
    eval_metrics_ndh_conv = get_evaluation_metrics(testing_GT_mask_2Hz_ndh, testing_gm_ndh_conv)
    eval_metrics_ndh_opt = get_evaluation_metrics(testing_GT_mask_2Hz_ndh, testing_gm_ndh_opt)
    eval_metrics_dh_conv = get_evaluation_metrics(testing_GT_mask_2Hz_dh, testing_gm_dh_conv)
    eval_metrics_dh_opt = get_evaluation_metrics(testing_GT_mask_2Hz_dh, testing_gm_dh_opt)
    eval_metrics_bil_conv = get_evaluation_metrics(testing_GT_mask_2Hz_bil, testing_gm_bil_conv)
    eval_metrics_bil_opt = get_evaluation_metrics(testing_GT_mask_2Hz_bil, testing_gm_bil_opt)

    # Create evaluation metrics dictionary
    eval_metrics = {
        'ndh_conv': eval_metrics_ndh_conv,
        'ndh_opt': eval_metrics_ndh_opt,
        'dh_conv': eval_metrics_dh_conv,
        'dh_opt': eval_metrics_dh_opt,
        'bil_conv': eval_metrics_bil_conv,
        'bil_opt': eval_metrics_bil_opt
    }

    return gm_scores, eval_metrics


def compute_evaluation_metrics_gmac(testing_pitch_mad_ndh, testing_pitch_mad_dh,
                                                       testing_count_brond_ndh, testing_count_brond_dh,
                                                       testing_GT_mask_ndh_1Hz, testing_GT_mask_dh_1Hz, testing_GT_mask_bil_1Hz,
                                                       opt_threshold_ndh, opt_threshold_dh, group_optimal_fs_ndh, group_optimal_fs_dh):
    """
    Compute evaluation metrics for different prediction scenarios using the GMAC algorithm.

    Args:
        testing_pitch_mad_ndh (numpy.ndarray): NumPy array containing pitch_mad values for the left bronchus (NDH).
        testing_pitch_mad_dh (numpy.ndarray): NumPy array containing pitch_mad values for the right bronchus (DH).
        testing_count_brond_ndh (numpy.ndarray): NumPy array containing AC values for the left bronchus (NDH).
        testing_count_brond_dh (numpy.ndarray): NumPy array containing AC values for the right bronchus (DH).
        testing_GT_mask_ndh_1Hz (pandas.Series): Series containing GT mask values for the left bronchus (NDH).
        testing_GT_mask_dh_1Hz (pandas.Series): Series containing GT mask values for the right bronchus (DH).
        testing_GT_mask_bil_1Hz (pandas.Series): Series containing bilateral GT mask values.
        opt_threshold_ndh (float): Optimal AC threshold for NDH prediction.
        opt_threshold_dh (float): Optimal AC threshold for DH prediction.
        group_optimal_fs_ndh (function): Group optimal functional space function for NDH prediction.
        group_optimal_fs_dh (function): Group optimal functional space function for DH prediction.

    Returns:
        A tuple containing two dictionaries:
        - GMAC scores dictionary
        - Evaluation metrics dictionary
    """
    # Compute GMAC predictions
    testing_gmac_ndh_conv = compute_GMAC(testing_pitch_mad_ndh, testing_count_brond_ndh, ac_threshold=0, functional_space=30)
    testing_gmac_ndh_opt = compute_GMAC(testing_pitch_mad_ndh, testing_count_brond_ndh, ac_threshold=opt_threshold_ndh, functional_space=group_optimal_fs_ndh)
    testing_gmac_dh_conv = compute_GMAC(testing_pitch_mad_dh, testing_count_brond_dh, ac_threshold=0, functional_space=30)
    testing_gmac_dh_opt = compute_GMAC(testing_pitch_mad_dh, testing_count_brond_dh, ac_threshold=opt_threshold_dh, functional_space=group_optimal_fs_dh)
    testing_gmac_bil_conv = get_mask_bilateral(testing_gmac_ndh_conv, testing_gmac_dh_conv)
    testing_gmac_bil_opt = get_mask_bilateral(testing_gmac_ndh_opt, testing_gmac_dh_opt)

    # Compute GMAC score dictionaries
    gmac_scores = {
        'ndh_conv': testing_gmac_ndh_conv,
        'ndh_opt': testing_gmac_ndh_opt,
        'dh_conv': testing_gmac_dh_conv,
        'dh_opt': testing_gmac_dh_opt,
        'bil_conv': testing_gmac_bil_conv,
        'bil_opt': testing_gmac_bil_opt
    }

    # Compute evaluation metrics
    eval_metrics_gmac_conv_ndh = get_evaluation_metrics(testing_GT_mask_ndh_1Hz, testing_gmac_ndh_conv)
    eval_metrics_gmac_opt_ndh = get_evaluation_metrics(testing_GT_mask_ndh_1Hz, testing_gmac_ndh_opt)
    eval_metrics_gmac_conv_dh = get_evaluation_metrics(testing_GT_mask_dh_1Hz, testing_gmac_dh_conv)
    eval_metrics_gmac_opt_dh = get_evaluation_metrics(testing_GT_mask_dh_1Hz, testing_gmac_dh_opt)
    eval_metrics_gmac_conv_bil = get_evaluation_metrics(testing_GT_mask_bil_1Hz, testing_gmac_bil_conv)
    eval_metrics_gmac_opt_bil = get_evaluation_metrics(testing_GT_mask_bil_1Hz, testing_gmac_bil_opt)

    # Create evaluation metrics dictionary
    eval_metrics = {
        'ndh_conv': eval_metrics_gmac_conv_ndh,
        'ndh_opt': eval_metrics_gmac_opt_ndh,
        'dh_conv': eval_metrics_gmac_conv_dh,
        'dh_opt': eval_metrics_gmac_opt_dh,
        'bil_conv': eval_metrics_gmac_conv_bil,
        'bil_opt': eval_metrics_gmac_opt_bil
    }

    return gmac_scores, eval_metrics


def plot_multiple_radar_plot(eval_metrics, figures_path, metric):
    """
    Plot multiple radar charts and bar charts based on evaluation metrics.

    Args:
        eval_metrics (dict): Dictionary containing evaluation metrics for different scenarios.
        metric (str): Name of the metric being plotted.
        figures_path (str or None): Path where the figures should be saved or None to not save.

    Returns:
        None.
    """

    # Function to build a save path or return None if figures_path is None
    def build_save_path(base_path, filename):
        if base_path is not None:
            return base_path + '/' + filename
        return None

    base_path = None if figures_path is None else figures_path + '/' + metric

    # Radar Plot
    # Plot radar chart for NDH scenario (conv vs opt)
    plot_radar_chart(eval_metrics['ndh_conv'], eval_metrics['ndh_opt'], metric, save_filename=build_save_path(base_path, metric + '_radar_NDH'))

    # Plot radar chart for DH scenario (conv vs opt)
    plot_radar_chart(eval_metrics['dh_conv'], eval_metrics['dh_opt'], metric, save_filename=build_save_path(base_path, metric + '_radar_DH'))

    # Plot radar chart for bilateral scenario (conv vs opt)
    plot_radar_chart(eval_metrics['bil_conv'], eval_metrics['bil_opt'], metric, save_filename=build_save_path(base_path, metric + '_radar_bil'))

    # BAR Plot
    plot_bar_chart(eval_metrics['ndh_conv'], eval_metrics['ndh_opt'], metric, save_filename=build_save_path(base_path, metric + '_bar_NDH'))
    plot_bar_chart(eval_metrics['dh_conv'], eval_metrics['dh_opt'], metric, save_filename=build_save_path(base_path, metric + '_bar_DH'))
    plot_bar_chart(eval_metrics['bil_conv'], eval_metrics['bil_opt'], metric, save_filename=build_save_path(base_path, metric + '_bar_bil'))

    
    
def plot_bar_chart(conventional_metrics, optimal_metrics, metric, save_filename=None):
    metric_names = list(conventional_metrics.keys())
    num_metrics = len(metric_names)
    bar_width = 0.35
    ind = np.arange(num_metrics)  # X-axis locations for bars
    
    # Load a bold font for annotations
    prop = fm.FontProperties(weight='bold')

    fig, ax = plt.subplots(figsize=(10, 8))

    # Extract the values from the metrics dictionaries
    conventional_values = list(conventional_metrics.values())
    optimal_values = list(optimal_metrics.values())

    # Plot the bars
    rects1 = ax.bar(ind, conventional_values, bar_width, label='Conventional', color='blue')
    rects2 = ax.bar(ind + bar_width, optimal_values, bar_width, label='Optimal', color='green')

    ax.set_xlabel('Metrics')
    ax.set_ylabel('Percentage')
    ax.set_title(f'Comparison of {metric} Metrics: Conventional vs Optimal')
    ax.set_xticks(ind + bar_width / 2)
    ax.set_xticklabels(metric_names)
    ax.legend()

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
    else:
        plt.tight_layout()
        plt.show()  # Show the plot if save_filename is not provided
        
        
def optimal_group_ac_threshold_computation(group_count_brond, group_GT_mask_1Hz, optimal=True):
    conventional_threshold_unilateral = 2
   
    # Ensure datasets have the same size 
    group_count_brond, group_GT_mask_1Hz = remove_extra_elements(group_count_brond, group_GT_mask_1Hz)
    if optimal:
        # Train your model and find the optimal threshold 
        optimal_threshold = find_optimal_threshold(group_GT_mask_1Hz, group_count_brond)
    else: 
        optimal_threshold = conventional_threshold_unilateral
        print('Using conventional threshold')
    
    return optimal_threshold


def optimal_group_fs_computation(group_pitch_mad_50Hz, group_yaw_mad_50Hz, group_GT_mask_50Hz, optimal = True): 
    
    conventional_fs = 30 # degrees
    optimal_thresholds = []
    
    # Set of angles to test 
    functional_space_array = list(range(5, 91, 2))
    
    # Ensure datasets have the same size 
    group_pitch_mad_50Hz, group_GT_mask_50Hz = remove_extra_elements(group_pitch_mad_50Hz, group_GT_mask_50Hz)
    group_yaw_mad_50Hz, group_GT_mask_50Hz = remove_extra_elements(group_yaw_mad_50Hz, group_GT_mask_50Hz)
    
    # Downsample GT mask from 50 Hz to 2 Hz 
    group_GT_mask_2Hz = resample_mask(group_GT_mask_50Hz, 50.0, 2.0)
        
    # Get optimal fs by finding the angle giving the gm score array having the highest Youden Index score when compared to the GT 
    if optimal:
        optimal_fs = test_fs_values(group_pitch_mad_50Hz, group_yaw_mad_50Hz, group_GT_mask_2Hz, functional_space_array)
    else:
        optimal_fs = conventional_fs
        
    return optimal_fs


def plot_side_by_side_boxplots(individual_optimal_threshold_ndh, individual_optimal_threshold_dh,
                               group_optimal_threshold_ndh, group_optimal_threshold_dh, metric):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    if metric == 'AC':
        conventional_threshold_unilateral = 2
        plot_title = 'Distribution of the AC optimal thresholds across individuals'
    elif metric == 'GM':
        conventional_threshold_unilateral = 30
        plot_title = 'Distribution of the optimal functional spaces across individuals'

    # Colors for 'ndh' and 'dh' sides
    ndh_color = 'skyblue'
    dh_color = 'lightgreen'

    # Box plot for ndh side
    ndh_box = plt.boxplot(individual_optimal_threshold_ndh, positions=[1], labels=['ndh'], patch_artist=True, boxprops=dict(facecolor=ndh_color))
    # Box plot for dh side
    dh_box = plt.boxplot(individual_optimal_threshold_dh, positions=[2], labels=['dh'], patch_artist=True, boxprops=dict(facecolor=dh_color))

    # Add the threshold line for the conventional threshold
    plt.axhline(y=conventional_threshold_unilateral, color='red', linestyle='--', label=f'Conventional threshold = {conventional_threshold_unilateral}')

    # Add the dashed lines for the optimal thresholds
    plt.axhline(y=group_optimal_threshold_ndh, color='blue', linestyle='--', label=f'Group Optimal NDH Threshold = {group_optimal_threshold_ndh:.2f}')
    plt.axhline(y=group_optimal_threshold_dh, color='green', linestyle='--', label=f'Group Optimal DH Threshold = {group_optimal_threshold_dh:.2f}')

    plt.title(plot_title)
    plt.xlabel('Side')
    plt.ylabel('Optimal Threshold')

    if metric == 'GM':
        plt.legend(loc='best')  # Adjust the legend location for better visibility
    else:
        plt.legend()

    plt.tight_layout()
    plt.show()

    # Calculate and print the average values
    avg_ndh = np.mean(individual_optimal_threshold_ndh)
    avg_dh = np.mean(individual_optimal_threshold_dh)
    print(f'Average ndh: {avg_ndh:.2f}')
    print(f'Average dh: {avg_dh:.2f}')

    # Calculate and print the median values
    median_ndh = np.median(individual_optimal_threshold_ndh)
    median_dh = np.median(individual_optimal_threshold_dh)
    print(f'Median ndh: {median_ndh:.2f}')
    print(f'Median dh: {median_dh:.2f}')

    
def get_testing_data(initial_path, testing_participant_id, frequency_GT, frequency_gm):
    testing_participant_path = os.path.join(initial_path, testing_participant_id)
    testing_data = load_testing_data(testing_participant_path)

    (testing_count_brond_ndh, testing_count_brond_dh,
     testing_GT_mask_ndh_1Hz, testing_GT_mask_dh_1Hz, testing_GT_mask_bil_1Hz,
     testing_pitch_mad_ndh, testing_yaw_mad_ndh, testing_pitch_mad_dh, testing_yaw_mad_dh,
     testing_GT_mask_50Hz_ndh, testing_GT_mask_50Hz_dh, testing_GT_mask_bil_50Hz) = testing_data

    # Resample the testing masks
    # Downsample the GT mask from 50 Hz to 2 Hz to allow comparison with GM scores
    testing_GT_mask_2Hz_ndh = resample_mask(testing_GT_mask_50Hz_ndh, frequency_GT, frequency_gm)
    testing_GT_mask_2Hz_dh = resample_mask(testing_GT_mask_50Hz_dh, frequency_GT, frequency_gm)
    testing_GT_mask_2Hz_bil = get_mask_bilateral(testing_GT_mask_2Hz_ndh, testing_GT_mask_2Hz_dh)

    return (testing_count_brond_ndh, testing_count_brond_dh,
            testing_GT_mask_ndh_1Hz, testing_GT_mask_dh_1Hz, testing_GT_mask_bil_1Hz,
            testing_pitch_mad_ndh, testing_yaw_mad_ndh, testing_pitch_mad_dh, testing_yaw_mad_dh,
            testing_GT_mask_50Hz_ndh, testing_GT_mask_50Hz_dh, testing_GT_mask_bil_50Hz,
            testing_GT_mask_2Hz_ndh, testing_GT_mask_2Hz_dh, testing_GT_mask_2Hz_bil)


def get_duration_functional_arm_use(scores_dict, sampling_frequency):
    """
    Calculate the duration and percentage of active epochs for functional arm use.

    Args:
        scores_dict (dict): A dictionary containing binary scores for 6 different fields.
        sampling_frequency (int): The sampling frequency of the data.

    Returns:
        dict: A dictionary containing duration and percentage of active epochs for each field.
    """
    metric_duration_arm_use = {}

    for field, scores in scores_dict.items():
        active_epochs = sum(scores)
        total_epochs = len(scores)

        percentage_active = (active_epochs / total_epochs) * 100

        duration_seconds = active_epochs / sampling_frequency
        duration_formatted = "{:02d}:{:02d}:{:02d}".format(
            int(duration_seconds // 3600),
            int((duration_seconds % 3600) // 60),
            int(duration_seconds % 60)
        )

        metric_duration_arm_use[field] = {
            'percentage_active': percentage_active,
            'duration_formatted': duration_formatted
        }

    return metric_duration_arm_use


def compare_arm_use_duration_plot(ground_truth, metric_duration, metric_name, save_path=None):
    """
    Compare arm use duration using bar charts.

    Args:
        ground_truth (dict): Ground truth arm use duration dictionary.
        metric_duration (dict): Metric duration arm use dictionary.
        metric_name (str): Name of the metric being compared ('AC', 'GM', or 'GMAC').
        save_path (str): Path to save the figure. If None, the figure won't be saved.

    Returns:
        None
    """
    sns.set(style="whitegrid")
    
    sides = ['ndh', 'dh', 'bil']

    for side in sides:
        plt.figure(figsize=(10, 6))
        plt.title(f"Functional Arm Use Duration Comparison - {side.capitalize()} - {metric_name}", fontsize=16)

        ground_truth_percentage = ground_truth[side]['percentage_active']
        metric_duration_conv_percentage = metric_duration[f'{side}_conv']['percentage_active']
        metric_duration_opt_percentage = metric_duration[f'{side}_opt']['percentage_active']

        x = [0, 1, 2]
        heights = [ground_truth_percentage, metric_duration_conv_percentage, metric_duration_opt_percentage]
        labels = ['Ground Truth', 'Conventional', 'Optimal']

        colors = sns.color_palette("Set1")

        ax = sns.barplot(x=x, y=heights, palette=colors)
        plt.xticks(x, labels)
        plt.ylabel('Percentage Active Duration (%)', fontsize=12)
        plt.xlabel('')
        
        # Display duration in the middle of each bar
        durations = [
            ground_truth[side]['duration_formatted'],
            metric_duration[f'{side}_conv']['duration_formatted'],
            metric_duration[f'{side}_opt']['duration_formatted']
        ]
        for i, duration in enumerate(durations):
            ax.text(i, heights[i] / 2, duration, ha='center', fontsize=10, fontweight='bold')
        
        # Calculate and display percentage differences
        diff_conv = ((metric_duration_conv_percentage - ground_truth_percentage) / ground_truth_percentage) * 100
        diff_opt = ((metric_duration_opt_percentage - ground_truth_percentage) / ground_truth_percentage) * 100
        
        # Display + or - value indicating the percentage difference on top of bars
        diff_conv_symbol = '+' if diff_conv >= 0 else '-'
        diff_opt_symbol = '+' if diff_opt >= 0 else '-'
        
        ax.text(1, metric_duration_conv_percentage + 1, f'{diff_conv_symbol}{abs(int(diff_conv))}%',
                ha='center', fontsize=12, fontweight='bold', color='black')
        
        ax.text(2, metric_duration_opt_percentage + 1, f'{diff_opt_symbol}{abs(int(diff_opt))}%',
                ha='center', fontsize=12, fontweight='bold', color='black')

        if save_path:
            file_name = f"Functional_Arm_Use_Duration_Comparison_{side.capitalize()}_{metric_name}.png"
            full_file_path = os.path.join(save_path, file_name)
            plt.savefig(full_file_path)
            print(f"Figure saved as '{full_file_path}'")
            plt.show()
        else:
            plt.show()










