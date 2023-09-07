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


def extract_values_across_participants(paths, *fields):
    """
    Extracts specific fields' values for each participant from their JSON files.

    Args:
    - paths (list): List of paths to the participants' JSON files.
    - *fields (str): The fields to extract from each JSON file.

    Returns:
    - tuple: Tuple containing numpy arrays of extracted values for the given fields across all participants.
    """
    
    values_dict = {field: [] for field in fields}

    for path in paths:
        with open(path, 'r') as file:
            data = json.load(file)
            for field in fields:
                if field in data:
                    values_dict[field].append(data[field])
                else:
                    values_dict[field].append(np.nan)

    # Convert lists in the dictionary to numpy arrays
    for field in fields:
        values_dict[field] = np.array(values_dict[field])

    return tuple(values_dict.values())


def get_group_data(participant_paths, field):
    """
    Get combined data for a group of participants for a specific field.
    
    Args:
    - participant_paths (list): List of file paths to JSON files for each participant.
    - field (str): The field to extract data for.
    
    Returns:
    - list: A merged dataset of equally trimmed data for the field from all participants.
    """
    
    all_data = []
    min_length = float('inf')
    
    # Step 1: Load JSON files and extract the relevant field data
    for path in participant_paths:
        with open(path, 'r') as f:
            data_dict = json.load(f)
            if field in data_dict:
                current_data = data_dict[field]
                all_data.append(current_data)
    
    # Step 2: Find the minimum length among all extracted arrays
    min_length = min(len(data) for data in all_data)
    
    # Step 3: Downsample each participant's data and store it
    resampled_group_data = []
    for data in all_data:
        original_frequency = len(data)
        desired_frequency = min_length  # Assuming min_length is the length of the smallest dataset

        # Skip resampling if the dataset is already at the desired frequency
        if original_frequency == desired_frequency:
            resampled_values = data
        else:
            if 'GT' in field:
                print('Mask detected')
                resampled_values = resample_binary_mask(data, original_frequency, desired_frequency)
            elif 'AC' in field:
                print('AC detected')
                resampled_values = resample_AC(data, original_frequency, desired_frequency)
            elif 'yaw' in field or 'pitch' in field:
                print('Angle detected')
                resampled_values = resample_angle_data(data, original_frequency, desired_frequency)
            else:
                raise ValueError("Invalid field type for resampling.")

        if len(resampled_values) != desired_frequency:
            raise ValueError(f"Resampling did not return expected length. Expected {desired_frequency}, got {len(resampled_values)}")
        
        # Plot and append data
        plot_resampled_arrays(data, original_frequency, resampled_values, desired_frequency)
        resampled_group_data.append(resampled_values)
        
    # Step 4: Merge the resampled data into a single dataset
    group_dataset = np.concatenate(resampled_group_data)
    
    return group_dataset


def compute_evaluation_metrics_single_case_ac(count_brond, GT_mask, thresholds, limb_condition):
    """
    Compute evaluation metrics for AC scores.
    
    Parameters:
    - count_brond: np.array | The count of BRoND
    - GT_mask: np.array | Ground Truth Mask
    - thresholds: dict | Dictionary containing the various thresholds
    - limb_condition: str | Specifies which limb condition is applicable ("DH", "NDH", or "BIL")
    
    Returns:
    Tuple containing two dictionaries:
    - ac_scores: Dictionary containing AC scores for conventional and optimal approaches
    - eval_metrics: Dictionary containing evaluation metrics for conventional and optimal approaches
    """
    
    # Determine threshold for conventional AC based on limb condition
    conv_threshold = thresholds['conventional_AC_threshold_bilateral'] if limb_condition == 'BIL' else thresholds['conventional_AC_threshold_unilateral']
    
    # Determine group AC threshold for optimal AC based on limb condition
    opt_threshold = thresholds['group_AC_NDH'] if limb_condition == 'NDH' else thresholds['group_AC_DH']
    
    # Calculate AC scores
    if limb_condition == 'BIL':
        ac_scores = {
            'conv': get_prediction_bilateral(count_brond['NDH'], thresholds['conventional_AC_threshold_unilateral'],
                                             count_brond['DH'], thresholds['conventional_AC_threshold_unilateral']),
            'opt': get_prediction_bilateral(count_brond['NDH'], thresholds['group_AC_NDH'],
                                            count_brond['DH'], thresholds['group_AC_DH'])
        }
    else:
        ac_scores = {
            'conv': get_prediction_ac(count_brond, conv_threshold),
            'opt': get_prediction_ac(count_brond, opt_threshold)
        }
    
    # Calculate evaluation metrics
    eval_metrics = {
        'conv': get_evaluation_metrics(GT_mask, ac_scores['conv']),
        'opt': get_evaluation_metrics(GT_mask, ac_scores['opt'])
    }
    
    return ac_scores, eval_metrics


def compute_evaluation_metrics_single_case_gm(pitch_mad, yaw_mad, functional_space, GT_mask_2Hz, limb_condition):
    """
    Compute evaluation metrics for GM scores.
    
    Parameters:
    - pitch_mad: np.array | Pitch MAD (Mean Angular Deviation?)
    - yaw_mad: np.array | Yaw MAD
    - functional_space: np.array or dict | Functional Space for GM calculation
    - GT_mask_2Hz: np.array | Ground Truth Mask at 2Hz
    - limb_condition: str | Specifies which limb condition is applicable ("DH", "NDH", or "BIL")
    
    Returns:
    Tuple containing two dictionaries:
    - gm_scores: Dictionary containing GM scores for conventional and optimal approaches
    - eval_metrics: Dictionary containing evaluation metrics for conventional and optimal approaches
    """
    
    # Calculate GM scores
    if limb_condition == 'BIL':
        gm_scores = {
            'conv': get_mask_bilateral(
                gm_algorithm(pitch_mad['NDH'], yaw_mad['NDH'], functional_space['conv']),
                gm_algorithm(pitch_mad['DH'], yaw_mad['DH'], functional_space['conv'])
            ),
            'opt': get_mask_bilateral(
                gm_algorithm(pitch_mad['NDH'], yaw_mad['NDH'], functional_space['opt_ndh']),
                gm_algorithm(pitch_mad['DH'], yaw_mad['DH'], functional_space['opt_dh'])
            )
        }
    else:
        gm_scores = {
            'conv': gm_algorithm(pitch_mad, yaw_mad, functional_space['conv']),
            'opt': gm_algorithm(pitch_mad, yaw_mad, functional_space['opt'])
        }
    
    # Calculate evaluation metrics
    eval_metrics = {
        'conv': get_evaluation_metrics(GT_mask_2Hz, gm_scores['conv']),
        'opt': get_evaluation_metrics(GT_mask_2Hz, gm_scores['opt'])
    }
    
    return gm_scores, eval_metrics


def compute_evaluation_metrics_single_case_gmac(pitch_mad, count_brond, GT_mask_1Hz, thresholds, limb_condition):
    """
    Compute evaluation metrics for GMAC scores.

    Parameters:
    - pitch_mad: np.array | Pitch MAD
    - count_brond: np.array | Count from AC metric
    - GT_mask_1Hz: np.array | Ground Truth Mask at 1Hz
    - thresholds: dict | Threshold values for GMAC computations
    - limb_condition: str | Specifies which limb condition is applicable ("DH", "NDH", or "BIL")

    Returns:
    Tuple containing two dictionaries:
    - gmac_scores: Dictionary containing GMAC scores for conventional and optimal approaches
    - eval_metrics: Dictionary containing evaluation metrics for conventional and optimal approaches
    """
    
    opt_threshold_ndh = thresholds['group_AC_NDH']
    opt_threshold_dh = thresholds['group_AC_DH']
    group_optimal_fs_ndh = thresholds['group_optimal_FS_NDH']
    group_optimal_fs_dh = thresholds['group_optimal_FS_DH']

    if limb_condition == 'BIL':
        gmac_scores = {
            'conv': get_mask_bilateral(
                compute_GMAC(pitch_mad['NDH'], count_brond['NDH'], ac_threshold=0, functional_space=30),
                compute_GMAC(pitch_mad['DH'], count_brond['DH'], ac_threshold=0, functional_space=30)
            ),
            'opt': get_mask_bilateral(
                compute_GMAC(pitch_mad['NDH'], count_brond['NDH'], ac_threshold=opt_threshold_ndh, functional_space=group_optimal_fs_ndh),
                compute_GMAC(pitch_mad['DH'], count_brond['DH'], ac_threshold=opt_threshold_dh, functional_space=group_optimal_fs_dh)
            )
        }
    elif limb_condition == 'NDH':
        gmac_scores = {
            'conv': compute_GMAC(pitch_mad, count_brond, ac_threshold=0, functional_space=30),
            'opt': compute_GMAC(pitch_mad, count_brond, ac_threshold=opt_threshold_ndh, functional_space=group_optimal_fs_ndh)
        }
    elif limb_condition == 'DH':
        gmac_scores = {
            'conv': compute_GMAC(pitch_mad, count_brond, ac_threshold=0, functional_space=30),
            'opt': compute_GMAC(pitch_mad, count_brond, ac_threshold=opt_threshold_dh, functional_space=group_optimal_fs_dh)
        }

    # Calculate evaluation metrics
    eval_metrics = {
        'conv': get_evaluation_metrics(GT_mask_1Hz, gmac_scores['conv']),
        'opt': get_evaluation_metrics(GT_mask_1Hz, gmac_scores['opt'])
    }
    
    return gmac_scores, eval_metrics


def compute_evaluation_metrics(participant_data, thresholds, metric):
    # Initialize dictionaries to hold metric scores and evaluation metrics
    metric_scores = {}
    metric_eval = {}

    if metric == 'AC':
        # Extract relevant data from the testing participant_data
        testing_count_brond_ndh = np.array(participant_data['AC_NDH'])
        testing_count_brond_dh = np.array(participant_data['AC_DH'])
        
        testing_GT_mask_ndh_1Hz = np.array(participant_data['GT_mask_NDH_1Hz'])
        testing_GT_mask_dh_1Hz = np.array(participant_data['GT_mask_DH_1Hz'])
        testing_GT_mask_bil_1Hz = get_mask_bilateral(testing_GT_mask_ndh_1Hz, testing_GT_mask_dh_1Hz)

        # Compute metrics for non-dominant hand (NDH)
        metric_scores['ndh'], metric_eval['ndh'] = compute_evaluation_metrics_single_case_ac(
            testing_count_brond_ndh, testing_GT_mask_ndh_1Hz, thresholds, "NDH")

        # Compute metrics for dominant hand (DH)
        metric_scores['dh'], metric_eval['dh'] = compute_evaluation_metrics_single_case_ac(
            testing_count_brond_dh, testing_GT_mask_dh_1Hz, thresholds, "DH")

        # Compute metrics for bilateral (BIL)
        metric_scores['bil'], metric_eval['bil'] = compute_evaluation_metrics_single_case_ac(
            {'NDH': testing_count_brond_ndh, 'DH': testing_count_brond_dh},
            testing_GT_mask_bil_1Hz, thresholds, "BIL")
        
    elif metric == 'GM':
        
        # Extract relevant data for GM metric from participant_data
        testing_pitch_mad_ndh = np.array(participant_data['pitch_NDH'])
        testing_yaw_mad_ndh = np.array(participant_data['yaw_NDH'])
        testing_pitch_mad_dh = np.array(participant_data['pitch_DH'])
        testing_yaw_mad_dh = np.array(participant_data['yaw_DH'])

        testing_GT_mask_2Hz_ndh = np.array(participant_data['GT_mask_NDH_2Hz'])
        testing_GT_mask_2Hz_dh = np.array(participant_data['GT_mask_DH_2Hz'])
        testing_GT_mask_2Hz_bil = get_mask_bilateral(testing_GT_mask_2Hz_ndh, testing_GT_mask_2Hz_dh)

        # Define functional spaces for GM calculations
        conventional_functional_space = thresholds['conventional_functional_space']
        group_optimal_fs_ndh = thresholds['group_optimal_FS_NDH']
        group_optimal_fs_dh = thresholds['group_optimal_FS_DH']

        # Compute metrics for non-dominant hand (NDH)
        metric_scores['ndh'], metric_eval['ndh'] = compute_evaluation_metrics_single_case_gm(
            testing_pitch_mad_ndh, testing_yaw_mad_ndh, 
            {'conv': conventional_functional_space, 'opt': group_optimal_fs_ndh}, 
            testing_GT_mask_2Hz_ndh, "NDH"
        )

        # Compute metrics for dominant hand (DH)
        metric_scores['dh'], metric_eval['dh'] = compute_evaluation_metrics_single_case_gm(
            testing_pitch_mad_dh, testing_yaw_mad_dh, 
            {'conv': conventional_functional_space, 'opt': group_optimal_fs_dh}, 
            testing_GT_mask_2Hz_dh, "DH"
        )

        # Compute metrics for bilateral (BIL)
        metric_scores['bil'], metric_eval['bil'] = compute_evaluation_metrics_single_case_gm(
            {'NDH': testing_pitch_mad_ndh, 'DH': testing_pitch_mad_dh}, 
            {'NDH': testing_yaw_mad_ndh, 'DH': testing_yaw_mad_dh}, 
            {'conv': conventional_functional_space, 'opt_ndh': group_optimal_fs_ndh, 'opt_dh': group_optimal_fs_dh}, 
            testing_GT_mask_2Hz_bil, "BIL"
        )
    
    elif metric == 'GMAC':
        
        # Extract the relevant data from the participant_data for both AC and GM metrics
        # (Assumes that these have been computed previously)
        testing_count_brond_ndh = np.array(participant_data['AC_NDH'])
        testing_count_brond_dh = np.array(participant_data['AC_DH'])
        testing_pitch_mad_ndh = np.array(participant_data['pitch_NDH'])
        testing_pitch_mad_dh = np.array(participant_data['pitch_DH'])

        testing_GT_mask_ndh_1Hz = np.array(participant_data['GT_mask_NDH_1Hz'])
        testing_GT_mask_dh_1Hz = np.array(participant_data['GT_mask_DH_1Hz'])
        testing_GT_mask_bil_1Hz = get_mask_bilateral(testing_GT_mask_ndh_1Hz, testing_GT_mask_dh_1Hz)

        # Compute metrics for non-dominant hand (NDH)
        metric_scores['ndh'], metric_eval['ndh'] = compute_evaluation_metrics_single_case_gmac(
            testing_pitch_mad_ndh, testing_count_brond_ndh, testing_GT_mask_ndh_1Hz, thresholds, "NDH")

        # Compute metrics for dominant hand (DH)
        metric_scores['dh'], metric_eval['dh'] = compute_evaluation_metrics_single_case_gmac(
            testing_pitch_mad_dh, testing_count_brond_dh, testing_GT_mask_dh_1Hz, thresholds, "DH")

        # Compute metrics for bilateral (BIL)
        metric_scores['bil'], metric_eval['bil'] = compute_evaluation_metrics_single_case_gmac(
            {'NDH': testing_pitch_mad_ndh, 'DH': testing_pitch_mad_dh},
            {'NDH': testing_count_brond_ndh, 'DH': testing_count_brond_dh},
            testing_GT_mask_bil_1Hz, thresholds, "BIL")
    
    return metric_scores, metric_eval


def plot_multiple_radar_plot(eval_metrics, figures_path, metric, show_plot=False):
    """
    Plot multiple radar charts and bar charts based on evaluation metrics.

    Args:
        eval_metrics (dict): Dictionary containing evaluation metrics for different scenarios.
        metric (str): Name of the metric being plotted.
        figures_path (str or None): Path where the figures should be saved, or None to not save.
        show_plot (bool): Whether to display the plot in the notebook.

    Returns:
        None.
    """

    def build_save_path(base_path, scenario, plot_type):
        if base_path is not None:
            return os.path.join(base_path, f"{metric}_{plot_type}_{scenario}.png")
        return None

    base_path = figures_path  # The base_path is simply the figures_path or None.
    
    if base_path and not os.path.exists(base_path):
        os.makedirs(base_path)  # Create directory if it doesn't exist

    # Loop through scenarios and types of plots
    for scenario in ['ndh', 'dh', 'bil']:
        for plot_type in ['radar', 'bar']:
            save_path = build_save_path(base_path, scenario, plot_type)
            
            if plot_type == 'radar':
                plot_radar_chart(eval_metrics[scenario]['conv'], eval_metrics[scenario]['opt'], metric, scenario,
                                 save_filename=save_path, show_plot=show_plot)
            else:
                plot_bar_chart(eval_metrics[scenario]['conv'], eval_metrics[scenario]['opt'], metric, scenario,
                               save_filename=save_path, show_plot=show_plot)
                
                
def plot_bar_chart(conventional_metrics, optimal_metrics, metric, scenario, save_filename=None, show_plot=True):
    
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
    ax.set_title(f'Comparison of {metric} Metrics: Conventional vs Optimal [Side: {scenario.upper()}]')
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
    if show_plot:
        plt.tight_layout()
        plt.show()  # Show the plot if show_plot is True

        
def configure_axis(ax, angles, metric_names):
    """
    Configure the axis for the radar chart.
    
    Args:
        ax (matplotlib.pyplot.axis): Axis object for the radar plot.
        angles (list): Angles for the metrics.
        metric_names (list): Names of the metrics being plotted.
        
    Returns:
        None.
    """
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names)

    
def plot_data(ax, angles, values, label, color):
    """
    Plot data on the radar chart.
    
    Args:
        ax (matplotlib.pyplot.axis): Axis object for the radar plot.
        angles (list): Angles for the metrics.
        values (list): Values of the metrics.
        label (str): Label for the data.
        color (str): Color for the plot.
        
    Returns:
        None.
    """
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=label, color=color)
    ax.fill(angles, values, alpha=0.50, color=color)

def annotate_points(ax, angles, values, metric_names):
    """
    Annotate the points on the radar chart.
    
    Args:
        ax (matplotlib.pyplot.axis): Axis object for the radar plot.
        angles (list): Angles for the metrics.
        values (list): Values of the metrics.
        metric_names (list): Names of the metrics being plotted.
        
    Returns:
        None.
    """
    for angle, value, metric_name in zip(angles, values, metric_names):
        ax.annotate(f"{round(value)}%", xy=(angle, value), xytext=(angle, value + 0.05),
                    horizontalalignment='center', verticalalignment='center')


def plot_radar_chart(conventional_metrics, optimal_metrics, metric, scenario, save_filename=None, show_plot=True):
    metric_names = list(conventional_metrics.keys())
    num_metrics = len(metric_names)

    angles = [n / float(num_metrics) * 2 * np.pi for n in range(num_metrics)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

    conventional_values = [round(conventional_metrics[metric_name]) for metric_name in metric_names]
    optimal_values = [round(optimal_metrics[metric_name]) for metric_name in metric_names]

    configure_axis(ax, angles, metric_names)

    label = f'Conventional {metric}' if metric != 'AC' else 'Conventional Threshold'
    plot_data(ax, angles, conventional_values, label, 'blue')
    annotate_points(ax, angles, conventional_values, metric_names)

    label = f'Optimal {metric}' if metric != 'AC' else 'Optimal Threshold'
    plot_data(ax, angles, optimal_values, label, 'green')
    annotate_points(ax, angles, optimal_values, metric_names)

    ax.set_title(f'Evaluation Metrics Comparison between Conventional vs Optimal {metric} [Side: {scenario.upper()}]')
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.tight_layout()
    
    if save_filename:
        plt.savefig(save_filename)
    if show_plot:
        plt.tight_layout()
        plt.show()


def get_duration_functional_arm_use(scores_dict, sampling_frequency):
    """
    Calculate the duration and percentage of active epochs for functional arm use.

    Args:
        scores_dict (dict): A nested dictionary containing binary scores for different methods and conditions.
        sampling_frequency (int): The sampling frequency of the data.

    Returns:
        dict: A dictionary containing duration and percentage of active epochs for each field.
    """
    metric_duration_arm_use = {}

    for limb, methods in scores_dict.items():
        metric_duration_arm_use[limb] = {}

        if not isinstance(methods, dict):
            print(f"Skipping '{limb}' as its associated value is not a dictionary.")
            continue

        for method, scores in methods.items():
            active_epochs = np.sum(scores)  # sum the numpy array to count the number of active epochs
            total_epochs = len(scores)

            percentage_active = (active_epochs / total_epochs) * 100

            duration_seconds = active_epochs / sampling_frequency
            duration_formatted = "{:02d}:{:02d}:{:02d}".format(
                int(duration_seconds // 3600),
                int((duration_seconds % 3600) // 60),
                int(duration_seconds % 60)
            )

            metric_duration_arm_use[limb][method] = {
                'percentage_active': percentage_active,
                'duration_formatted': duration_formatted
            }

    return metric_duration_arm_use


def get_GT_dict(testing_dataset):
    """
    Constructs a dictionary containing ground truth masks for NDH, DH, and BIL at 50Hz.

    Args:
        testing_dataset (dict): Participant dataset containing GT_mask_DH_50Hz and GT_mask_NDH_50Hz.

    Returns:
        dict: Dictionary containing ground truth masks for NDH, DH, and BIL at 50Hz.
    """
    
    # Initialize dictionary to store ground truth masks
    testing_dict_GT_mask_25Hz = {}
    
    # Initialize sub-dictionaries for each limb condition to store the ground truth mask for 'GT'
    testing_dict_GT_mask_50Hz['NDH'] = {'GT': testing_dataset['GT_mask_NDH_25Hz']}
    testing_dict_GT_mask_50Hz['DH'] = {'GT': testing_dataset['GT_mask_DH_25Hz']}
    
    # Compute the ground truth mask for bilateral (BIL) condition
    bil_gt_mask = get_mask_bilateral(testing_dataset['GT_mask_NDH_25Hz'], testing_dataset['GT_mask_DH_25Hz'])
    testing_dict_GT_mask_25Hz['BIL'] = {'GT': bil_gt_mask}

    return testing_dict_GT_mask_25Hz


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
    
    # Note: Changing 'sides' to uppercase to match the ground_truth keys
    sides = ['NDH', 'DH', 'BIL']
    for side in sides:
        plt.figure(figsize=(10, 6))
        plt.title(f"Functional Arm Use Duration Comparison - {side} - {metric_name}", fontsize=16)

        ground_truth_percentage = ground_truth[side]['GT']['percentage_active']
        metric_duration_conv_percentage = metric_duration[side.lower()]['conv']['percentage_active']
        metric_duration_opt_percentage = metric_duration[side.lower()]['opt']['percentage_active']

        x = [0, 1, 2]
        heights = [ground_truth_percentage, metric_duration_conv_percentage, metric_duration_opt_percentage]
        labels = ['Ground Truth', 'Conventional', 'Optimal']

        colors = sns.color_palette("Set1")

        ax = sns.barplot(x=x, y=heights, palette=colors)
        plt.xticks(x, labels)
        plt.ylabel('Percentage Active Duration (%)', fontsize=12)
        plt.xlabel('')
        
        durations = [
            ground_truth[side]['GT']['duration_formatted'],
            metric_duration[side.lower()]['conv']['duration_formatted'],
            metric_duration[side.lower()]['opt']['duration_formatted']
        ]
        
        for i, duration in enumerate(durations):
            ax.text(i, heights[i] / 2, duration, ha='center', fontsize=10, fontweight='bold')
        
         # Calculate and display percentage differences
        diff_conv = ((metric_duration_conv_percentage - ground_truth_percentage) / ground_truth_percentage) * 100
        diff_opt = ((metric_duration_opt_percentage - ground_truth_percentage) / ground_truth_percentage) * 100
        
        # Display + or - value indicating the percentage difference on top of bars
        diff_conv_symbol = '+' if diff_conv >= 0 else '-'
        diff_opt_symbol = '+' if diff_opt >= 0 else '-'

        ax.text(1, metric_duration_conv_percentage + 1, f"Deviation from GT: {diff_conv_symbol}{abs(int(diff_conv))}%", 
                ha='center', fontsize=12, fontweight='bold', color='black')
        
        ax.text(2, metric_duration_opt_percentage + 1, f"Deviation from GT: {diff_opt_symbol}{abs(int(diff_opt))}%", 
                ha='center', fontsize=12, fontweight='bold', color='black')

        if save_path:
            file_name = f"Functional_Arm_Use_Duration_Comparison_{side}_{metric_name}.png"
            full_file_path = os.path.join(save_path, file_name)
            plt.savefig(full_file_path)
            print(f"Figure saved as '{full_file_path}'")
            plt.show()
        else:
            plt.show()