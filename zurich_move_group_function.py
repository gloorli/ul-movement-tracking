import os
import pandas as pd
import numpy as np 
from scipy.interpolate import CubicSpline
from activity_count_function import *

def find_files_with_extension(file_extension):
    """
    Searches the current directory for folders starting with 'H' and finds the associated
    files with the given extension inside each folder.

    Args:
        file_extension (str): The extension of the files to search for.

    Returns:
        List of paths to files with the given extension.
    """
    current_dir = os.getcwd()

    # Use list comprehension for a cleaner code
    csv_files = [os.path.join(folder, file_extension)
                 for folder in os.listdir(current_dir)
                 if folder.startswith('H') and os.path.isdir(folder)
                 and os.path.isfile(os.path.join(folder, file_extension))]

    return csv_files


def get_resampled_acc(trimmed_data_csv_files):
    all_acc_data = []

    # Find the shortest length among all the arrays
    min_length = float('inf')

    for trimmed_data in trimmed_data_csv_files:
        # Convert to a dataframe
        df = pd.read_csv(trimmed_data)

        # Keep only the fields: acc_x, acc_y, acc_z
        acc_data = df[['acc_x', 'acc_y', 'acc_z']].values

        # Update the minimum length
        min_length = min(min_length, len(acc_data))

        all_acc_data.append(acc_data)

    # Cubic spline interpolation to match the smaller array size
    resampled_acc_data = []

    for acc_data in all_acc_data:
        # Perform cubic spline interpolation
        x = np.arange(len(acc_data))
        cs = CubicSpline(x, acc_data)
        resampled_acc_data.append(cs(np.linspace(0, len(acc_data) - 1, min_length)))

    return resampled_acc_data


def regroup_field_data(csv_files):
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


def get_metrics_list(group_metrics, threshold_label, metric_label, side_labels):
    field_names = [f'{threshold_label}_{side_label}_{metric_label}' for side_label in side_labels]
    data_list = [group_metrics[field_name] for field_name in field_names]
    title = f'{threshold_label} for {metric_label}'
    return data_list, title


def plot_vertical_whisker(data_list, title):
    """
    Plots multiple vertical whisker plots to represent multiple datasets.

    Args:
        data_list: List of numpy arrays or lists of data values.
        title: Title for the plot.

    Returns:
        None (displays the plot).
    """

    plt.figure(figsize=(10, 6))  # Adjust the figure size as desired

    positions = np.arange(len(data_list))

    # Plot the vertical whisker plots
    labels = ['aff.', 'nonaff.', 'bilat.']
    plt.boxplot(data_list, positions=positions, vert=True, labels=labels)

    # Set labels and title with larger font size
    plt.xlabel('Data', fontsize=14)
    plt.ylabel('Values', fontsize=14)
    plt.title(title, fontsize=16)

    # Calculate the lower limit dynamically based on the minimum value of the data
    min_value = min(np.min(data) for data in data_list)
    lower_limit = max(0, min_value - 10)  # Add a buffer of 10 to the minimum value

    # Set the y-axis limits dynamically
    plt.ylim(lower_limit, 100)

    # Customize the style
    plt.style.use('ggplot')  # Use the 'ggplot' style or choose another style

    # Show the plot
    plt.show()
