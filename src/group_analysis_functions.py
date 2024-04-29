# Standard Library Imports
import os
import csv

# Third-Party Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.interpolate import CubicSpline
from adjustText import adjust_text

# Local Imports
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
        avg_eval_metrics, optimal_threshold = k_fold_cross_validation(group_count_brond, group_GT_mask_1Hz)
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


def plot_side_by_side_boxplots_both_group(
        ind_opt_thres_h_ndh, ind_opt_thres_h_dh,
        ind_opt_thres_s_na, ind_opt_thres_s_a,
        grp_opt_thres_h_ndh, grp_opt_thres_h_dh,
        grp_opt_thres_s_na, grp_opt_thres_s_a,
        metric, path=None):

    sns.set(style="whitegrid")
    plt.figure(figsize=(14, 6))

    if metric == 'AC':
        conventional_threshold_unilateral = 2
        threshold_type = 'AC Threshold'
    elif metric == 'GM':
        conventional_threshold_unilateral = 30
        threshold_type = 'Functional Space'

    # New order of labels and data to match your preferred order
    positions = [1, 2, 4, 5]
    labels_legend_grp_thresh = ['Grp H-Dom', 'Grp H-NonDom', 'Grp S-NonAff', 'Grp S-Aff']
    labels = ['Healthy Dom. H', 'Healthy Non-Dom. H', 'Stroke Non-Aff. H', 'Stroke Aff. H']
    all_data = [ind_opt_thres_h_dh, ind_opt_thres_h_ndh, ind_opt_thres_s_na, ind_opt_thres_s_a]
    grp_opt_thres = [grp_opt_thres_h_dh, grp_opt_thres_h_ndh, grp_opt_thres_s_na, grp_opt_thres_s_a]

    # The order of box and threshold colors should also be re-arranged
    box_colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightpink']
    grp_thres_colors = ['deepskyblue', 'limegreen', 'indianred', 'hotpink']

    handles = []
    for pos, data, box_color, label, grp_thres, grp_color in zip(positions, all_data, box_colors, labels, grp_opt_thres, grp_thres_colors):
        plt.boxplot(data, positions=[pos], labels=[label], patch_artist=True, boxprops=dict(facecolor=box_color))
        hline = plt.hlines(y=grp_thres, xmin=pos-0.2, xmax=pos+0.2, colors=grp_color)
        handles.append(hline)

    plt.axhline(y=conventional_threshold_unilateral, color='red', linestyle='--', label=f'Conventional {threshold_type} = {conventional_threshold_unilateral}')
    plt.title(f'Distribution of the Optimal {threshold_type}s', fontsize=16)
    plt.xlabel('Groups and Sides', fontsize=14)
    plt.xticks(fontsize=14)  # Change 16 to your desired font size
    plt.ylabel(f'Individual Optimal {threshold_type}', fontsize=14)
    plt.yticks(fontsize=14)  # Change 16 to your desired font size

    y_max = max([max(data) for data in all_data] + [conventional_threshold_unilateral])
    plt.ylim([0, y_max * 1.2])

    # Create legend for group thresholds
    grp_legend_handles = [Line2D([0], [0], color=grp_thres_color, linewidth=2) for grp_thres_color in grp_thres_colors]
    grp_legend_labels = [f"{label} ({grp_thres})" for label, grp_thres in zip(labels_legend_grp_thresh, grp_opt_thres)]  # using labels_legend_grp_thresh instead of labels


    # Create legend handle for the conventional threshold
    conventional_legend_handle = Line2D([0], [0], color='red', linewidth=2, linestyle='--')
    conventional_legend_label = f"Conventional {threshold_type} = {conventional_threshold_unilateral}"

    # Combine all legend handles and labels
    all_legend_handles = grp_legend_handles + [conventional_legend_handle]
    all_legend_labels = grp_legend_labels + [conventional_legend_label]

    # Create the legend
    plt.legend(handles=all_legend_handles, labels=all_legend_labels, loc='best', title='Legend')
    plt.tight_layout()

    if path is not None:
        filename = f'boxplot_{metric}.png'
        full_path = f"{path}/{filename}"
        plt.savefig(full_path)
        print(f"Figure saved at {full_path}")

    plt.show()


def plot_side_metrics(data_array_h, data_array_s, metric_names, algorithm, save_path=None):
    for metric_name in metric_names:
        # Initialize arrays for each group ('H' and 'S')
        ndh_data_ot_h, ndh_data_ct_h, dh_data_ot_h, dh_data_ct_h = [], [], [], []
        ndh_data_ot_s, ndh_data_ct_s, dh_data_ot_s, dh_data_ct_s = [], [], [], []
        
        # Data extraction for 'H' group
        for data in data_array_h:
            for key, value in data.items():
                parts = key.split('_')
                if len(parts) == 3 and parts[2] == metric_name:
                    if parts[0] == 'OT' and parts[1] == 'ndh':
                        ndh_data_ot_h.append(value)
                    elif parts[0] == 'CT' and parts[1] == 'ndh':
                        ndh_data_ct_h.append(value)
                    elif parts[0] == 'OT' and parts[1] == 'dh':
                        dh_data_ot_h.append(value)
                    elif parts[0] == 'CT' and parts[1] == 'dh':
                        dh_data_ct_h.append(value)

        # Data extraction for 'S' group
        for data in data_array_s:
            for key, value in data.items():
                parts = key.split('_')
                if len(parts) == 3 and parts[2] == metric_name:
                    if parts[0] == 'OT' and parts[1] == 'ndh':
                        ndh_data_ot_s.append(value)
                    elif parts[0] == 'CT' and parts[1] == 'ndh':
                        ndh_data_ct_s.append(value)
                    elif parts[0] == 'OT' and parts[1] == 'dh':
                        dh_data_ot_s.append(value)
                    elif parts[0] == 'CT' and parts[1] == 'dh':
                        dh_data_ct_s.append(value)

        # Check data availability
        if not (ndh_data_ot_h and ndh_data_ct_h and dh_data_ot_h and dh_data_ct_h and 
                ndh_data_ot_s and ndh_data_ct_s and dh_data_ot_s and dh_data_ct_s):
            print(f"Data not found for the metric: {metric_name}")
            continue

        # Call the plotting function
        plot_data_side_by_side(
            ndh_data_ct_h, ndh_data_ot_h, dh_data_ct_h, dh_data_ot_h,
            ndh_data_ct_s, ndh_data_ot_s, dh_data_ct_s, dh_data_ot_s,
            metric_name, algorithm, save_path
        )


# Function to plot boxplots for both 'H' and 'S' groups
def plot_data_side_by_side(
        data1_ct_h, data1_ot_h, data2_ct_h, data2_ot_h,
        data1_ct_s, data1_ot_s, data2_ct_s, data2_ot_s,
        metric_name, algorithm, save_path):
    
    sns.set_style("whitegrid")
    plt.figure(figsize=(16, 6))

    # Positions and labels
    positions = np.arange(1, 17, 2)
    width = 0.4
    labels_NDH_H = ['H Non-Dom.', 'H Non-Dom.']
    labels_DH_H = ['H Dom.', 'H Dom.']
    labels_NDH_S = ['S Aff.', 'S Aff.']
    labels_DH_S = ['S Non-Aff.', 'S Non-Aff.']
    all_labels = labels_NDH_H + labels_DH_H + labels_NDH_S + labels_DH_S
    all_data = [data1_ct_h, data1_ot_h, data2_ct_h, data2_ot_h, data1_ct_s, data1_ot_s, data2_ct_s, data2_ot_s]

    plt.boxplot(all_data, positions=positions, labels=all_labels, patch_artist=True, widths=width, whis=2)
    
    colors = ['lightblue', 'lightgreen'] * 4
    for patch, color in zip(plt.gca().patches, colors):
        patch.set_facecolor(color)
    
    plt.title(r'Distribution of the ' + metric_name + ' for ' + algorithm + 
          r' Metric Across $\bf{Healthy}$ and $\bf{Stroke}$ Individuals (CT vs OT)', fontsize=16)

    plt.xlabel('Side and Group', fontsize=14)
    
    plt.ylabel(metric_name, fontsize=14)
    plt.xticks(fontsize=14) 
    plt.yticks(fontsize=14) 
    
    legend_elements = [plt.Rectangle((0, 0), 1, 1, color='lightblue', label='Conventional Parameters'),
                       plt.Rectangle((0, 0), 1, 1, color='lightgreen', label='Optimal Parameters')]
    
    plt.legend(handles=legend_elements, loc='best', prop={'size': 12})
    
    if save_path is not None:
        filename = f'boxplot_across_individuals_{algorithm}_H_S_{metric_name}.png'
        full_path = f"{save_path}/{filename}"
        plt.savefig(full_path)
        print(f"Figure saved at {full_path}")

    plt.show()

    
def get_group_optimal_thresholds(path):
    csv_files = [
        'H_optimal_threshold_AC.csv', 
        'S_optimal_threshold_AC.csv',
        'H_optimal_threshold_GM.csv',
        'S_optimal_threshold_GM.csv'
    ]

    H_AC_ndh, H_AC_dh, S_AC_ndh, S_AC_dh = 0, 0, 0, 0
    H_GM_ndh, H_GM_dh, S_GM_ndh, S_GM_dh = 0, 0, 0, 0

    for csv_file in csv_files:
        file_path = f"{path}/{csv_file}"
        df = pd.read_csv(file_path)
        
        # Extract the 'Threshold' values and map them based on 'Side'
        thresholds = df.set_index('Side')['Threshold'].to_dict()

        if csv_file == 'H_optimal_threshold_AC.csv':
            H_AC_ndh, H_AC_dh = thresholds['ndh'], thresholds['dh']
        elif csv_file == 'S_optimal_threshold_AC.csv':
            S_AC_ndh, S_AC_dh = thresholds['ndh'], thresholds['dh']
        elif csv_file == 'H_optimal_threshold_GM.csv':
            H_GM_ndh, H_GM_dh = thresholds['ndh'], thresholds['dh']
        elif csv_file == 'S_optimal_threshold_GM.csv':
            S_GM_ndh, S_GM_dh = thresholds['ndh'], thresholds['dh']

    return H_AC_ndh, H_AC_dh, S_AC_ndh, S_AC_dh, H_GM_ndh, H_GM_dh, S_GM_ndh, S_GM_dh


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


def compare_arm_use_duration_plot(ground_truth, metric_duration, metric_name, group, save_path=None):
    sns.set(style="white")
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    
    sides = ['NDH', 'DH']
    
    if group == 'H':
        plotting_side = ['Non-Dom. H', 'Dom. H']
    else: 
        plotting_side = ['Aff. H', 'Non-Aff. H']
    
    # Create a figure and a 1x2 subplot grid (one row, two columns)
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    for ax, side, plot_side in zip(reversed(axes), sides, plotting_side):

        ax.set_title(f"Functional Arm Use Duration Comparison - {plot_side} - {metric_name}", fontsize=18, pad=20)
        
        ground_truth_percentage = ground_truth[side]['GT']['percentage_active']
        metric_duration_conv_percentage = metric_duration[side.lower()]['conv']['percentage_active']
        metric_duration_opt_percentage = metric_duration[side.lower()]['opt']['percentage_active']
        
        x = [0, 1, 2]
        heights = [ground_truth_percentage, metric_duration_conv_percentage, metric_duration_opt_percentage]
        labels = ['Ground Truth', 'Conventional', 'Optimal']
        colors = ['#808080', '#007BFF', '#28a745']
        
        sns.barplot(x=x, y=heights, palette=colors, edgecolor=".2", ax=ax)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Percentage Active Duration (%)', fontsize=14)
        ax.set_xlabel('')
        ax.set_yticks(np.arange(0, 101, 10))
        
        durations = [
            ground_truth[side]['GT']['duration_formatted'],
            metric_duration[side.lower()]['conv']['duration_formatted'],
            metric_duration[side.lower()]['opt']['duration_formatted']
        ]
        
        for i, duration in enumerate(durations):
            ax.text(i, heights[i] / 2, duration, ha='center', fontsize=12, fontweight='bold')
        
        reference_val = ground_truth_percentage
        diff_conv = ((metric_duration_conv_percentage - reference_val) / reference_val) * 100
        diff_opt = ((metric_duration_opt_percentage - reference_val) / reference_val) * 100
        diff_conv_symbol = '+' if diff_conv >= 0 else '-'
        diff_opt_symbol = '+' if diff_opt >= 0 else '-'
        
        y_max = max(heights)
        ax.set_ylim(0, y_max + y_max * 0.2)
        
        ax.text(0, ground_truth_percentage + y_max * 0.01, "Reference", ha='center', fontsize=12, fontweight='bold', color='black')
        ax.text(1, metric_duration_conv_percentage + y_max * 0.01, f"Deviation from GT : {diff_conv_symbol}{abs(int(diff_conv))}%", ha='center', fontsize=12, fontweight='bold', color='black')
        ax.text(2, metric_duration_opt_percentage + y_max * 0.01, f"Deviation from GT : {diff_opt_symbol}{abs(int(diff_opt))}%", ha='center', fontsize=12, fontweight='bold', color='black')
        
    if save_path:
        file_name = f"Functional_Arm_Use_Duration_Comparison_{metric_name}.png"
        full_file_path = os.path.join(save_path, file_name)
        plt.savefig(full_file_path)
        print(f"Figure saved as '{full_file_path}'")
    else:
        plt.show()


def build_save_path_with_participant_id(base_path, participant_id, metric):
    """
    Constructs a save path for the plot by appending the participant ID folder to the base path.

    Parameters:
    base_path (str): The base directory where the plots should be saved.
    participant_id (str): The ID of the participant.
    metric (str): The metric for which the plot is being created.

    Returns:
    str: The full path where the plot should be saved.
    """
    # Append folder named after testing participant ID to the base path
    path_eval_fig = os.path.join(base_path, f"testing_{participant_id}")
    
    # Create directory if it doesn't exist
    os.makedirs(path_eval_fig, exist_ok=True)
    
    # Construct the full save path
    save_path = os.path.join(path_eval_fig, f"{metric}_combined.png")
    
    return save_path


def plot_combined_evaluation_metrics(eval_metrics, figures_path, participant_id, metric, group, show_plot=True):
    
    # Saving path 
    if figures_path is not None: 
        save_path = build_save_path_with_participant_id(figures_path, participant_id, metric)
    
    # Initialize plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.set(style="whitegrid")
    
    scenarios = ['dh', 'ndh']
    metric_names = ['Sensitivity', 'Specificity', 'Accuracy']
    
    # Adding space between groups by adjusting the positions of the second group
    ind = np.array([0, 1, 2, 4, 5, 6])
    bar_width = 0.35
    
    # Font property for bold text
    prop = fm.FontProperties(weight='bold')

    # Variables for legend
    legends_added = {'Conventional': False, 'Optimal': False}
    
    # Create bar plots
    for i, scenario in enumerate(scenarios):
        for j, m_name in enumerate(metric_names):
            position = ind[i * len(metric_names) + j]
            conv_value = eval_metrics[scenario]['conv'][m_name]
            opt_value = eval_metrics[scenario]['opt'][m_name]
            
            label_conv = 'Conventional' if not legends_added['Conventional'] else ""
            label_opt = 'Optimal' if not legends_added['Optimal'] else ""
            
            ax.bar(position - bar_width / 2, conv_value, bar_width, color='royalblue', label=label_conv)
            ax.bar(position + bar_width / 2, opt_value, bar_width, color='forestgreen', label=label_opt)
            
            legends_added['Conventional'] = True
            legends_added['Optimal'] = True
            
            # Annotate bars with percentage rounded to closest int
            for value, pos in [(conv_value, position - bar_width / 2), (opt_value, position + bar_width / 2)]:
                ax.annotate(f"{int(round(value))}%", 
                            xy=(pos, value), 
                            xytext=(0, 3), 
                            textcoords="offset points", 
                            ha='center', 
                            va='bottom', 
                            fontproperties=prop)
                
    ax.set_xticks(ind)
    
    if group == 'H':
        xticklabels = [f"{m} ({'Dom. H' if scenario == 'dh' else 'Non-Dom. H'})" for scenario in scenarios for m in metric_names]
    else:  # group == 'S'
        xticklabels = [f"{m} ({'Non-Aff. H' if scenario == 'dh' else 'Aff. H'})" for scenario in scenarios for m in metric_names]
    
    ax.set_ylim(0, 1.2 * max([eval_metrics[scenario]['conv'][m_name] for scenario in scenarios for m_name in metric_names] +
                             [eval_metrics[scenario]['opt'][m_name] for scenario in scenarios for m_name in metric_names]))

        
    ax.set_xticklabels(xticklabels, rotation=45)
    ax.set_xlabel('Evaluation Metrics and Sides', fontsize=14)
    plt.xticks(fontsize=14)
    ax.set_ylabel('Evaluation [%]', fontsize=14)
    ax.set_title(f'{metric} Comparison: Conventional vs Optimal', fontsize=14)
    ax.legend()
    
    # Place legend at best location
    ax.legend(loc='best')
    
    # Ensure there's enough room at the bottom for the x-axis labels
    fig.subplots_adjust(bottom=0.3)

    if figures_path is not None:
        plt.savefig(save_path)
        print(f'Figure saved at {save_path}')

    if show_plot:
        plt.tight_layout()
        plt.show()


def compute_testing_participant(testing_participant_paths, testing_group, initial_path): 
    # Get group dataset from the testing participants
    AC_NDH = get_group_data(testing_participant_paths, field = 'AC_NDH')
    AC_DH = get_group_data(testing_participant_paths, field = 'AC_DH')
    GT_mask_NDH_1Hz = get_group_data(testing_participant_paths, field = 'GT_mask_NDH_1Hz')
    GT_mask_DH_1Hz = get_group_data(testing_participant_paths, field = 'GT_mask_DH_1Hz')

    group_GT_mask_NDH_50Hz = get_group_data(testing_participant_paths, field='GT_mask_NDH_50Hz')
    group_GT_mask_DH_50Hz = get_group_data(testing_participant_paths, field='GT_mask_DH_50Hz')
    group_GT_mask_NDH_25Hz = get_group_data(testing_participant_paths, field='GT_mask_NDH_25Hz')
    group_GT_mask_DH_25Hz = get_group_data(testing_participant_paths, field='GT_mask_DH_25Hz')
    group_pitch_NDH = get_group_data(testing_participant_paths, field='pitch_NDH')
    group_pitch_DH = get_group_data(testing_participant_paths, field='pitch_DH')
    group_yaw_NDH = get_group_data(testing_participant_paths, field='yaw_NDH')
    group_yaw_DH = get_group_data(testing_participant_paths, field='yaw_DH')
    
    # Downsample @ 2 Hz 
    GT_frequency = 25 # Hz
    frequency_GM = 2 # Hz
    GT_mask_NDH_2Hz = resample_mask(group_GT_mask_NDH_25Hz, GT_frequency, frequency_GM)
    GT_mask_DH_2Hz = resample_mask(group_GT_mask_DH_25Hz, GT_frequency, frequency_GM)
    
    # Create a dictionary to save the merged_testing_participant
    merged_testing_dataset = {
        'participant_id': testing_group + '_merged', 
        'AC_NDH': AC_NDH,
        'AC_DH': AC_DH,
        'GT_mask_NDH_1Hz': GT_mask_NDH_1Hz,
        'GT_mask_DH_1Hz': GT_mask_DH_1Hz,
        'GT_mask_NDH_50Hz': group_GT_mask_NDH_50Hz,
        'GT_mask_DH_50Hz': group_GT_mask_DH_50Hz,
        'GT_mask_NDH_25Hz': group_GT_mask_NDH_25Hz,
        'GT_mask_DH_25Hz': group_GT_mask_DH_25Hz,
        'pitch_NDH': group_pitch_NDH,
        'pitch_DH': group_pitch_DH,
        'yaw_NDH': group_yaw_NDH,
        'yaw_DH': group_yaw_DH,
        'GT_mask_NDH_2Hz': GT_mask_NDH_2Hz, 
        'GT_mask_DH_2Hz':GT_mask_DH_2Hz
    }
    
    # Construct the path where the JSON file will be saved
    path_to_save = os.path.join(initial_path, f"{merged_testing_dataset['participant_id']}")
    
    # Save the dictionary as a JSON file
    save_to_json(merged_testing_dataset, path_to_save)
    
    # Print to indicate where the file was saved
    print(f"Dataset has been saved at: {path_to_save}")

    return merged_testing_dataset


def get_GT_dict(testing_dataset):
    """
    Constructs a dictionary containing ground truth masks for NDH, DH, and BIL at 25Hz.

    Args:
        testing_dataset (dict): Participant dataset containing GT_mask_DH_25Hz and GT_mask_NDH_25Hz.

    Returns:
        dict: Dictionary containing ground truth masks for NDH, DH, and BIL at 25Hz.
    """
    
    # Initialize dictionary to store ground truth masks
    testing_dict_GT_mask_25Hz = {}
    
    # Initialize sub-dictionaries for each limb condition to store the ground truth mask for 'GT'
    testing_dict_GT_mask_25Hz['NDH'] = {'GT': testing_dataset['GT_mask_NDH_25Hz']}
    testing_dict_GT_mask_25Hz['DH'] = {'GT': testing_dataset['GT_mask_DH_25Hz']}
    
    # Compute the ground truth mask for bilateral (BIL) condition
    bil_gt_mask = get_mask_bilateral(testing_dataset['GT_mask_NDH_25Hz'], testing_dataset['GT_mask_DH_25Hz'])
    testing_dict_GT_mask_25Hz['BIL'] = {'GT': bil_gt_mask}

    return testing_dict_GT_mask_25Hz
