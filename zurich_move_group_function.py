import os
import pandas as pd
import numpy as np 
from scipy.interpolate import CubicSpline
from activity_count_function import *
import csv
import matplotlib.pyplot as plt
import seaborn as sns


def find_specific_csv_files(initial_path, csv_file_names):
    """
    Searches for specific CSV files inside folders starting with 'H' within the 'CreateStudy' directory and its subdirectories.

    Args:
        initial_path (str): The path to the 'CreateStudy' directory.
        csv_file_names (list): A list of CSV file names to search for.

    Returns:
        A dictionary containing lists of paths to CSV files for each requested file name.
    """
    csv_files_dict = {csv_name: [] for csv_name in csv_file_names}

    # Walk through the directory tree starting from initial_path
    for root, dirs, files in os.walk(initial_path):
        for dir_name in dirs:
            # Check if the folder starts with 'H'
            if dir_name.startswith('H'):
                folder_path = os.path.join(root, dir_name)
                # Find the specific CSV files inside each folder
                for csv_name in csv_file_names:
                    csv_file_path = os.path.join(folder_path, csv_name)
                    if os.path.isfile(csv_file_path):
                        csv_files_dict[csv_name].append(csv_file_path)

    return csv_files_dict


def plot_side_metrics(data_dict, metric_names):
    for metric_name in metric_names:
        lw_data_ot = []
        lw_data_ct = []
        rw_data_ot = []
        rw_data_ct = []
        bilateral_data_ot = []
        bilateral_data_ct = []

        for key, value in data_dict.items():
            parts = key.split('_')
            if len(parts) == 3 and parts[2] == metric_name:
                if parts[0] == 'OT' and parts[1] == 'LW':
                    lw_data_ot.extend(value)
                elif parts[0] == 'CT' and parts[1] == 'LW':
                    lw_data_ct.extend(value)
                elif parts[0] == 'OT' and parts[1] == 'RW':
                    rw_data_ot.extend(value)
                elif parts[0] == 'CT' and parts[1] == 'RW':
                    rw_data_ct.extend(value)
                elif parts[0] == 'OT' and parts[1] == 'bilateral':
                    bilateral_data_ot.extend(value)
                elif parts[0] == 'CT' and parts[1] == 'bilateral':
                    bilateral_data_ct.extend(value)

        if not lw_data_ot or not lw_data_ct or not rw_data_ot or not rw_data_ct or not bilateral_data_ot or not bilateral_data_ct:
            print(f"Data not found for the metric: {metric_name}")
            continue
        # Plotting
        plot_data_side_by_side(lw_data_ct, lw_data_ot, rw_data_ct, rw_data_ot, bilateral_data_ct, bilateral_data_ot, metric_name)


def plot_data_side_by_side(data1_ct, data1_ot, data2_ct, data2_ot, data3_ct, data3_ot, metric_name):
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))

    positions = [1, 2, 4, 5, 7, 8]
    width = 0.2

    # Plot 'LW' side
    plt.boxplot([data1_ct, data1_ot], positions=positions[:2], labels=['LW CT', 'LW OT'], patch_artist=True, widths=width)
    # Plot 'RW' side
    plt.boxplot([data2_ct, data2_ot], positions=positions[2:4], labels=['RW CT', 'RW OT'], patch_artist=True, widths=width)
    # Plot 'bilateral' side
    plt.boxplot([data3_ct, data3_ot], positions=positions[4:], labels=['Bilateral CT', 'Bilateral OT'], patch_artist=True, widths=width)

    plt.title(f'{metric_name} Comparison for LW, RW, and Bilateral (CT vs OT)')
    plt.xlabel('Side')
    plt.ylabel(metric_name)

    colors = ['lightblue', 'lightgreen', 'lightblue', 'lightgreen', 'lightblue', 'lightgreen']
    for patch, color in zip(plt.gca().patches, colors):
        patch.set_facecolor(color)

    plt.tight_layout()
    plt.show()


def plot_side_by_side_boxplots(optimal_threshold_LW, optimal_threshold_RW, threshold, plot_title):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Colors for 'LW' and 'RW' sides
    lw_color = 'skyblue'
    rw_color = 'lightgreen'

    # Box plot for LW side
    plt.boxplot(optimal_threshold_LW, positions=[1], labels=['LW'], patch_artist=True, boxprops=dict(facecolor=lw_color))
    # Box plot for RW side
    plt.boxplot(optimal_threshold_RW, positions=[2], labels=['RW'], patch_artist=True, boxprops=dict(facecolor=rw_color))

    # Add the threshold line
    plt.axhline(y=threshold, color='red', linestyle='--', label=f'Conventional threshold = {threshold}')

    plt.title(plot_title)
    plt.xlabel('Side')
    plt.ylabel('Optimal Threshold')
    plt.legend()
    plt.tight_layout()
    plt.show()


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
    left_values = []
    right_values = []

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
                if side == 'left':
                    left_values.append(float(row[threshold_column]))
                elif side == 'right':
                    right_values.append(float(row[threshold_column]))

    return left_values, right_values


def get_group_data_from_csv(csv_files, mask=False):
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

    for csv_file in csv_files:
        # Read the CSV file and extract the data
        df = pd.read_csv(csv_file)
        data = df.iloc[:, 0].values
        # Update the minimum length
        min_length = min(min_length, len(data))
        all_data.append(data)

    # Cubic spline interpolation to match the smaller array size
    resampled_data = []
    
    for data in all_data:
        # Perform cubic spline interpolation
        x = np.arange(len(data))
        cs = CubicSpline(x, data, extrapolate=False)  # Set extrapolate to False
        resampled_values = cs(np.linspace(0, len(data) - 1, min_length))

        # Apply constraint to prevent negative values
        resampled_values = np.maximum(resampled_values, 0)  # Set negative values to zero

        # Calculate the number of elements removed
        num_removed = len(data) - len(resampled_values)
        elements_removed.append(num_removed)

        # Round the interpolated values to the nearest integer if it is a mask
        if mask:
            resampled_values = np.round(resampled_values).astype(int)
        resampled_data.append(resampled_values)
        
    group_data = np.concatenate(resampled_data, axis=0)

    # Print the number of elements removed for each array
    for idx, num_removed in enumerate(elements_removed, start=1):
        print(f"Elements removed in array {idx}: {num_removed}")

    return group_data










