import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score, jaccard_score
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d
from gm_function import *
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, f1_score, jaccard_score, recall_score, confusion_matrix
from sklearn.metrics import cohen_kappa_score
from utilities import *
from sklearn.model_selection import KFold


def get_data(folder):
    # Load IMU data
    LW_data_filename = 'trimmed_LW_data.csv'
    chest_data_filename = 'trimmed_chest_data.csv'
    RW_data_filename = 'trimmed_RW_data.csv'

    LW_data_path = os.path.join(folder, LW_data_filename)
    chest_data_path = os.path.join(folder, chest_data_filename)
    RW_data_path = os.path.join(folder, RW_data_filename)

    LW_data = pd.read_csv(LW_data_path)
    chest_data = pd.read_csv(chest_data_path)
    RW_data = pd.read_csv(RW_data_path)

    # Load Video GT data
    GT_mask_LW_filename = 'GT_mask_LW.csv'
    GT_mask_RW_filename = 'GT_mask_RW.csv'

    GT_mask_LW_path = os.path.join(folder, GT_mask_LW_filename)
    GT_mask_RW_path = os.path.join(folder, GT_mask_RW_filename)

    GT_mask_LW = pd.read_csv(GT_mask_LW_path)
    GT_mask_RW = pd.read_csv(GT_mask_RW_path)

    # Return the loaded dataframes
    return LW_data, chest_data, RW_data, GT_mask_LW, GT_mask_RW


def get_sampling_frequency(df):
    """
    Calculate the sampling frequency based on the time difference between the first and last timestamp in a DataFrame.

    Parameters:
    df (DataFrame): Input DataFrame containing the 'timestamp' column.

    Returns:
    float: Sampling frequency in Hertz.
    """
    # Convert 'timestamp' column to datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Get the first and last timestamps
    first_timestamp = df['timestamp'].iloc[0]
    last_timestamp = df['timestamp'].iloc[-1]

    # Calculate the time difference in seconds
    time_diff = (last_timestamp - first_timestamp).total_seconds()

    # Calculate the number of data points
    num_data_points = len(df)

    # Calculate the sampling frequency
    sampling_frequency = num_data_points / time_diff

    return sampling_frequency


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


def calculate_tpr_fpr(ground_truth, activity_counts, thresholds):
    tpr = []
    fpr = []
    
    for threshold in thresholds:
        # Apply the threshold to the activity counts
        predictions = activity_counts >= threshold
        
        # Calculate True Positive Rate (TPR) and False Positive Rate (FPR)
        tp = np.sum((ground_truth == 1) & (predictions == 1))
        fn = np.sum((ground_truth == 1) & (predictions == 0))
        tn = np.sum((ground_truth == 0) & (predictions == 0))
        fp = np.sum((ground_truth == 0) & (predictions == 1))
        
        tpr.append(tp / (tp + fn))
        fpr.append(fp / (fp + tn))
    
    return np.array(fpr), np.array(tpr)


def compute_bilateral_magnitude(activity_counts_sensor1, activity_counts_sensor2):
    # Check if the activity counts arrays have the same length
    if len(activity_counts_sensor1) != len(activity_counts_sensor2):
        raise ValueError("Activity counts arrays must have the same length.")

    # Compute the bilateral magnitude by summing the activity counts from both sensors
    bilateral_magnitude = []
    for count1, count2 in zip(activity_counts_sensor1, activity_counts_sensor2):
        bilateral_magnitude.append(count1 + count2)

    return np.array(bilateral_magnitude)


def get_magnitude_ratio(activity_counts_sensor_lw, activity_counts_sensor_rw):
    
    activity_counts_sensor_lw = np.array(activity_counts_sensor_lw)
    activity_counts_sensor_rw = np.array(activity_counts_sensor_rw)
    
    bilateral_magnitude = compute_bilateral_magnitude(activity_counts_sensor_lw, activity_counts_sensor_rw)
    
    # Create masks
    bm_zero_mask = (bilateral_magnitude == 0.0)
    lw_zero_mask = (activity_counts_sensor_lw == 0.0)
    rw_zero_mask = (activity_counts_sensor_rw == 0.0)
    
    # Compute the vector magnitude ratio for each second for valid division values
    mag_ratio = np.divide(activity_counts_sensor_lw, activity_counts_sensor_rw, out=np.ones_like(activity_counts_sensor_lw), where=(activity_counts_sensor_rw != 0) & (activity_counts_sensor_lw != 0))
    
    # Transform the ratio values using a natural logarithm only for the valid candidates
    mag_ratio_log = np.log(mag_ratio)
    
    # Handle cases with bilateral_magnitude = 0 
    mag_ratio_log[bm_zero_mask] = np.nan
    
    # Handle case with both lw_zero_mask AND rw_zero_mask active
    mag_ratio_log[np.logical_and(lw_zero_mask, rw_zero_mask)] = np.nan
    
    # Handle cases with lw_zero_mask OR rw_zero_mask active but not BOTH at the same time
    mag_ratio_log[np.logical_and(lw_zero_mask, ~rw_zero_mask)] = -7  # Left is at 0, indicating full use of the dominant (left) side 
    mag_ratio_log[np.logical_and(~lw_zero_mask, rw_zero_mask)] = 7   # Right is at 0, indicating full use of the non-dominant (right) side
    
    return mag_ratio_log


def get_tendency_ratio(ratio_array):
    """Calculate the percentage of values in an array of ratios between -7 and 0 (excluded),
       and between 0 (excluded) and +7. NaN values are excluded from the analysis.

    Args:
        ratio_array (array-like): Array of ratio values between -7 and 7.

    Returns:
        tuple: A tuple containing the percentage of values between -7 and 0 excluded, 
               and the percentage of values between 0 excluded and +7.
    """
    ratio_array = np.array(ratio_array)
    ratio_array = ratio_array[~np.isnan(ratio_array)] # remove NaN values
    
    neg_count = 0
    pos_count = 0
    total_count = len(ratio_array)
    
    for ratio in ratio_array:
        if ratio > 0:
            pos_count += 1
        elif ratio < 0:
            neg_count += 1
    
    neg_pct = round(neg_count / total_count * 100, 2)
    pos_pct = round(pos_count / total_count * 100, 2)
    zer_pct = round(100 - neg_pct - pos_pct, 2)
    
    return (neg_pct, pos_pct, zer_pct)


def plot_distribution_ratio(data):
    data = np.array(data)
    data = data[~np.isnan(data)] # remove NaN values

    plt.hist(data, bins=15, color='#1f77b4')
    plt.xlabel('Ratio [a.u.]')
    plt.ylabel('Frequency')
    plt.text(0.1, -0.15, 'Dominant side', transform=plt.gca().transAxes, horizontalalignment='center')
    plt.text(0.98, -0.15, 'Non Dominant Side', transform=plt.gca().transAxes, horizontalalignment='center', ha='right')
    
    # Add vertical line for median
    median_val = np.nanmedian(data)
    plt.axvline(median_val, linestyle='--', color='red', label='Median')
    plt.legend(loc='upper center')
    
    plt.show()


def plot_density(BM, ratio):
    # Define the range of the ratio values
    ratio_range = (-8, 8)
    
    ratio = np.squeeze(ratio)
    # Remove NaN values from the ratio and BM arrays
    nan_mask = np.isnan(ratio)
    BM = BM[~nan_mask]
    ratio = ratio[~nan_mask]
    
    # Determine the duration of each ratio value
    ratio_duration = np.zeros(len(ratio))
    unique_ratios = np.unique(ratio)
    for r in unique_ratios:
        mask = (ratio == r)
        ratio_duration[mask] = np.sum(mask)
    max_duration = np.max(ratio_duration)
    
    # Create the colormap for the density plot
    cmap = ListedColormap(plt.cm.get_cmap('bwr')(np.linspace(0, 1, 256)))
    
    # Create the density plot
    fig, ax = plt.subplots(figsize=(8, 6))
    img = ax.scatter(ratio, BM, c=ratio_duration, cmap=cmap, edgecolors='none', alpha=0.75)
    ax.set_xlim(ratio_range)
    ax.set_xlabel('Ratio')
    ax.set_ylabel('Bilateral Magnitude')
    #ax.set_title('Density Plot of Ratio and Bilateral Magnitude')
    plt.text(0.1, -0.15, 'Dominant side', transform=plt.gca().transAxes, horizontalalignment='center')
    plt.text(0.98, -0.15, 'Non Dominant Side', transform=plt.gca().transAxes, horizontalalignment='center', ha='right')
    
    # Create the color bar
    cbar = fig.colorbar(img)
    cbar.ax.set_ylabel('Duration (s)', rotation=270, labelpad=15)
    cbar.set_ticks(np.linspace(0, max_duration, 5, dtype=int))
    
    plt.show()


def get_mask_bilateral(GT_mask_LW, GT_mask_RW):
    """
    Creates a bilateral mask by performing element-wise logical AND operation on the given left and right masks.
    
    Args:
        GT_mask_LW (ndarray): Array representing the ground truth left mask.
        GT_mask_RW (ndarray): Array representing the ground truth right mask.
        
    Returns:
        ndarray: Bilateral mask where the value is 1 if and only if GT_mask_LW AND GT_mask_RW row is 1; otherwise, it's 0.
    """
    # Check if the input arrays have the same shape
    assert GT_mask_LW.shape == GT_mask_RW.shape, "The input arrays must have the same shape."
    
    # Perform element-wise logical AND operation on the masks
    mask_bilateral = np.logical_and(GT_mask_LW, GT_mask_RW).astype(int)
    
    return mask_bilateral


def save_evaluation_metrics_to_csv(df, folder_name):
    """
    Saves a DataFrame to a CSV file in the specified folder location.

    Args:
        df: DataFrame to be saved.
        folder_name: Name of the folder where the CSV file will be saved.

    Returns:
        None
    """
    # Ensure the folder exists, otherwise raise an error
    if not os.path.exists(folder_name):
        raise ValueError(f"The folder '{folder_name}' does not exist.")

    # Create the filename
    filename = 'evaluation_metrics.csv'

    # Construct the file path
    file_path = os.path.join(folder_name, filename)

    # Save the DataFrame to the CSV file
    df.to_csv(file_path, index=False)

    print(f"DataFrame saved to: {file_path}")


def save_AC_as_csv(count_brond_LW, count_brond_RW, folder):
    """
    Save AC values as CSV files.

    Args:
        count_brond_LW (DataFrame): DataFrame containing AC values for the left bronchus.
        count_brond_RW (DataFrame): DataFrame containing AC values for the right bronchus.
        folder (str): Folder path to save the CSV files.

    Returns:
        None.
    """
    # Specify the output CSV file names
    lw_output_filename = 'count_brond_LW.csv'
    rw_output_filename = 'count_brond_RW.csv'

    # Construct the full file paths for the output CSV files
    lw_output_path = os.path.join(folder, lw_output_filename)
    rw_output_path = os.path.join(folder, rw_output_filename)

    # Ensure the folder exists, otherwise raise an error
    if not os.path.exists(folder):
        raise ValueError(f"The folder '{folder}' does not exist.")

    # Save the AC values as CSV files
    count_brond_LW['AC Brond'].to_csv(lw_output_path, index=False)
    count_brond_RW['AC Brond'].to_csv(rw_output_path, index=False)

    # Print the path where the CSV files were saved
    if os.path.exists(lw_output_path) and os.path.exists(rw_output_path):
        print(f"CSV files saved successfully.")
        print(f"Left-hand AC CSV saved at: {lw_output_path}")
        print(f"Right-hand AC CSV saved at: {rw_output_path}")
    else:
        print(f"Failed to save CSV files.")


def get_prediction_bilateral(AC_LW, threshold_LW, AC_RW, threshold_RW):
    """
    Computes the bilateral prediction array based on the input arrays and thresholds.

    Args:
        AC_LW: Numpy array of values for left side.
        threshold_LW: Threshold value for dichotomization of left side.
        AC_RW: Numpy array of values for right side.
        threshold_RW: Threshold value for dichotomization of right side.

    Returns:
        Numpy array of bilateral predictions (0s and 1s).
    """
    # Generate predictions for left and right sides
    LW_pred = get_prediction(AC_LW, threshold_LW)
    RW_pred = get_prediction(AC_RW, threshold_RW)

    # Apply AND logic between the two arrays LW_pred and RW_pred
    pred_bilateral = np.logical_and(LW_pred, RW_pred)

    return pred_bilateral


def create_metrics_dictionary(metrics_LW_CT, metrics_RW_CT, metrics_bilateral_CT,
                             metrics_LW_OT, metrics_RW_OT, metrics_bilateral_OT):
    """
    Creates a dictionary with metrics data organized by the combination of vertical and horizontal axes.

    Args:
        metrics_LW_CT: Metrics data for LW and CT.
        metrics_RW_CT: Metrics data for RW and CT.
        metrics_bilateral_CT: Metrics data for bilateral and CT.
        metrics_LW_OT: Metrics data for LW and OT.
        metrics_RW_OT: Metrics data for RW and OT.
        metrics_bilateral_OT: Metrics data for bilateral and OT.

    Returns:
        Dictionary with metrics data organized by the combination of vertical and horizontal axes.
    """
    data = {
        ('OT_LW', 'Sensitivity'): metrics_LW_OT['Sensitivity'],
        ('OT_LW', 'Specificity'): metrics_LW_OT['Specificity'],
        ('OT_LW', 'Accuracy'): metrics_LW_OT['Accuracy'],
        ('OT_LW', 'PPV'): metrics_LW_OT['PPV'],
        ('OT_LW', 'NPV'): metrics_LW_OT['NPV'],
        ('OT_LW', 'F1 Score'): metrics_LW_OT['F1 Score'],
        ('OT_LW', 'Youden Index'): metrics_LW_OT['Youden Index'],
        ('OT_LW', 'False Positive Rate'): metrics_LW_OT['False Positive Rate'],
        ('OT_LW', 'False Negative Rate'): metrics_LW_OT['False Negative Rate'],
        ('OT_RW', 'Sensitivity'): metrics_RW_OT['Sensitivity'],
        ('OT_RW', 'Specificity'): metrics_RW_OT['Specificity'],
        ('OT_RW', 'Accuracy'): metrics_RW_OT['Accuracy'],
        ('OT_RW', 'PPV'): metrics_RW_OT['PPV'],
        ('OT_RW', 'NPV'): metrics_RW_OT['NPV'],
        ('OT_RW', 'F1 Score'): metrics_RW_OT['F1 Score'],
        ('OT_RW', 'Youden Index'): metrics_RW_OT['Youden Index'],
        ('OT_RW', 'False Positive Rate'): metrics_RW_OT['False Positive Rate'],
        ('OT_RW', 'False Negative Rate'): metrics_RW_OT['False Negative Rate'],
        ('OT_bilateral', 'Sensitivity'): metrics_bilateral_OT['Sensitivity'],
        ('OT_bilateral', 'Specificity'): metrics_bilateral_OT['Specificity'],
        ('OT_bilateral', 'Accuracy'): metrics_bilateral_OT['Accuracy'],
        ('OT_bilateral', 'PPV'): metrics_bilateral_OT['PPV'],
        ('OT_bilateral', 'NPV'): metrics_bilateral_OT['NPV'],
        ('OT_bilateral', 'F1 Score'): metrics_bilateral_OT['F1 Score'],
        ('OT_bilateral', 'Youden Index'): metrics_bilateral_OT['Youden Index'],
        ('OT_bilateral', 'False Positive Rate'): metrics_bilateral_OT['False Positive Rate'],
        ('OT_bilateral', 'False Negative Rate'): metrics_bilateral_OT['False Negative Rate'],
        ('CT_LW', 'Sensitivity'): metrics_LW_CT['Sensitivity'],
        ('CT_LW', 'Specificity'): metrics_LW_CT['Specificity'],
        ('CT_LW', 'Accuracy'): metrics_LW_CT['Accuracy'],
        ('CT_LW', 'PPV'): metrics_LW_CT['PPV'],
        ('CT_LW', 'NPV'): metrics_LW_CT['NPV'],
        ('CT_LW', 'F1 Score'): metrics_LW_CT['F1 Score'],
        ('CT_LW', 'Youden Index'): metrics_LW_CT['Youden Index'],
        ('CT_LW', 'False Positive Rate'): metrics_LW_CT['False Positive Rate'],
        ('CT_LW', 'False Negative Rate'): metrics_LW_CT['False Negative Rate'],
        ('CT_RW', 'Sensitivity'): metrics_RW_CT['Sensitivity'],
        ('CT_RW', 'Specificity'): metrics_RW_CT['Specificity'],
        ('CT_RW', 'Accuracy'): metrics_RW_CT['Accuracy'],
        ('CT_RW', 'PPV'): metrics_RW_CT['PPV'],
        ('CT_RW', 'NPV'): metrics_RW_CT['NPV'],
        ('CT_RW', 'F1 Score'): metrics_RW_CT['F1 Score'],
        ('CT_RW', 'Youden Index'): metrics_RW_CT['Youden Index'],
        ('CT_RW', 'False Positive Rate'): metrics_RW_CT['False Positive Rate'],
        ('CT_RW', 'False Negative Rate'): metrics_RW_CT['False Negative Rate'],
        ('CT_bilateral', 'Sensitivity'): metrics_bilateral_CT['Sensitivity'],
        ('CT_bilateral', 'Specificity'): metrics_bilateral_CT['Specificity'],
        ('CT_bilateral', 'Accuracy'): metrics_bilateral_CT['Accuracy'],
        ('CT_bilateral', 'PPV'): metrics_bilateral_CT['PPV'],
        ('CT_bilateral', 'NPV'): metrics_bilateral_CT['NPV'],
        ('CT_bilateral', 'F1 Score'): metrics_bilateral_CT['F1 Score'],
        ('CT_bilateral', 'Youden Index'): metrics_bilateral_CT['Youden Index'],
        ('CT_bilateral', 'False Positive Rate'): metrics_bilateral_CT['False Positive Rate'],
        ('CT_bilateral', 'False Negative Rate'): metrics_bilateral_CT['False Negative Rate'],
    }

    return data


def remove_wbm_data(mask_array, metric_array):
    # Remove the mask and metric values corresponding to the whole body movement epochs
    # Avoid bias in the Non-Functional data 
    
    wbm_label = -1
    
    filtered_mask = []
    filtered_metric = []

    for mask_value, metric_value in zip(mask_array, metric_array):
        if mask_value != wbm_label:
            filtered_mask.append(mask_value)
            filtered_metric.append(metric_value)
            
    filtered_mask = np.array(filtered_mask)
    filtered_metric = np.array(filtered_metric)
    return filtered_mask, filtered_metric


def save_resampled_masks_as_csv(GT_mask_LW_1Hz, GT_mask_RW_1Hz, folder):
    """
    Save masks as CSV files.

    Args:
        GT_mask_LW_1Hz (DataFrame): DataFrame containing the left-hand masks.
        GT_mask_RW_1Hz (DataFrame): DataFrame containing the right-hand masks.
        folder (str): Folder path to save the CSV files.

    Returns:
        None.
    """
    
    # Convert to DataFrame
    df_LW = pd.DataFrame(GT_mask_LW_1Hz, columns=['mask'])
    df_RW = pd.DataFrame(GT_mask_RW_1Hz, columns=['mask'])
    
    # Specify the output CSV file names
    lw_output_filename = 'GT_mask_LW_1Hz.csv'
    rw_output_filename = 'GT_mask_RW_1Hz.csv'

    # Construct the full file paths for the output CSV files
    lw_output_path = os.path.join(folder, lw_output_filename)
    rw_output_path = os.path.join(folder, rw_output_filename)

    # Ensure the folder exists, otherwise raise an error
    if not os.path.exists(folder):
        raise ValueError(f"The folder '{folder}' does not exist.")

    # Save the trimmed data
    df_LW.to_csv(lw_output_path, index=False)
    df_RW.to_csv(rw_output_path, index=False)

    # Print the path where the CSV files were saved
    if os.path.exists(lw_output_path) and os.path.exists(rw_output_path):
        print(f"CSV files saved successfully.")
        print(f"Left-hand mask CSV saved at: {lw_output_path}")
        print(f"Right-hand mask CSV saved at: {rw_output_path}")
    else:
        print(f"Failed to save CSV files.")


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


def replace_wbm_with_nf(mask):
    # Replace WBM mask by NF mask 
    mask = np.where(mask == -1, 0, mask)    
    return mask


def plot_boxplot_ac(ac_functional, ac_nf, ac_wbm):
    # Create a vertical box plot on the same figure with the 3 datasets represented as 3 box plots
    # y-axis is activity count

    # Set up the figure and axes
    plt.figure(figsize=(10, 6))

    # Combine the three datasets into a list to plot them on the same boxplot
    data = [ac_functional, ac_nf, ac_wbm]

    # Create three boxplots side by side using plt.boxplot()
    sns.boxplot(data=data, palette='Set3', width=0.6)

    # Set the x-axis labels
    plt.xticks(ticks=[0, 1, 2], labels=['Functional', 'Non-Functional', 'WBM'])

    # Set the y-axis label
    plt.ylabel('Activity Count')

    # Add a title
    plt.title('Boxplot of Activity Count')

    # Adjust the layout to avoid the x-axis labels being cut off
    plt.tight_layout()

    # Show the plot
    plt.show()

    
def plot_ac_tendency(mask_array, AC_array):
    AC_array = np.array(AC_array)
    
    # Create the three arrays of data using the mask
    ac_functional = AC_array[mask_array == 1]
    ac_nf = AC_array[mask_array == 0]
    ac_wbm = AC_array[mask_array == -1]

    # Call the function to plot the vertical box plot using the three arrays of data
    plot_boxplot_ac(ac_functional, ac_nf, ac_wbm) 


def compute_similarity_metric(ground_truth, gm_scores, metric):
    if metric == 'accuracy':
        return accuracy_score(ground_truth, gm_scores)
    elif metric == 'precision':
        return precision_score(ground_truth, gm_scores, zero_division=1.0) # Same as PPV
    elif metric == 'f1':
        return f1_score(ground_truth, gm_scores)
    elif metric == 'jaccard':
        return jaccard_score(ground_truth, gm_scores)
    elif metric == 'sensitivity':
        return recall_score(ground_truth, gm_scores)  # Same as recall
    elif metric == 'specificity':
        tn, fp, fn, tp = confusion_matrix(ground_truth, gm_scores).ravel()  # Same as TNR
        return tn / (tn + fp)
    elif metric == 'npv':
        tn, fp, fn, tp = confusion_matrix(ground_truth, gm_scores).ravel()
        return tn / (tn + fn)
    elif metric == 'cohen-kappa':
        return cohen_kappa_score(ground_truth, gm_scores)
    else:
        raise ValueError("Invalid similarity metric: " + metric)

    
def find_optimal_functional_space_gm(training_ground_truth, training_pitch, training_yaw, similarity_metrics):
    # Using training data only 
    functional_space_array = list(range(5, 91, 5))
    print("Functional Spaces tested:", functional_space_array)
    
    optimal_functional_spaces = {}
    optimal_functional_space_frequencies = {}
    
    for similarity_metric in similarity_metrics:
        print("Metric tested:", similarity_metric)
    
        best_similarity_score = -1
        optimal_functional_space = None
    
        for functional_space in functional_space_array:
        
            # Compute gm scores @ 2Hz using training pitch, yaw, and the given functional space
            gm_scores = gm_algorithm(training_pitch, training_yaw, functional_space=functional_space)
            
            # Ensure the two arrays have the same size
            gm_scores, training_ground_truth = remove_extra_elements(gm_scores, training_ground_truth)
            
            # Compute a similarity score between gm scores at this given functional space and the training ground truth dataset
            similarity_score = compute_similarity_metric(training_ground_truth, gm_scores, similarity_metric)
        
            # Update the similarity score and the new 'best' functional space if needed
            if similarity_score > best_similarity_score:
                best_similarity_score = similarity_score
                optimal_functional_space = functional_space
    
        # Store the optimal functional space and its corresponding similarity metric score
        optimal_functional_spaces[similarity_metric] = (optimal_functional_space, best_similarity_score)
        
        # Count the frequency of each optimal functional space value
        if optimal_functional_space in optimal_functional_space_frequencies:
            optimal_functional_space_frequencies[optimal_functional_space] += 1
        else:
            optimal_functional_space_frequencies[optimal_functional_space] = 1
    
    # Find the most frequent optimal functional space value (in case of equality, take the bigger one)
    most_frequent_optimal_space = max(optimal_functional_space_frequencies.keys())
    
    # Print the most frequent optimal functional space value
    print("Most Frequent Optimal Functional Space:", most_frequent_optimal_space)
    
    # Return the dictionary containing the optimal functional spaces for each similarity metric
    # and the most frequent optimal functional space value
    return optimal_functional_spaces, most_frequent_optimal_space


def visualize_metrics(metrics_dict):
    metric_names = list(metrics_dict.keys())
    values = [metric[1] * 100 for metric in metrics_dict.values()]
    optimal_functional_values = [metric[0] for metric in metrics_dict.values()]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(metric_names, values, color='skyblue')
    ax.set_xlabel('Metric')
    ax.set_ylabel('Value (%)')

    ax.set_ylim(0, max(values) + 5)  # Set the y-axis limit to accommodate labels on top of bars

    for i, v in enumerate(values):
        ax.text(i, v + 0.5, f'{v:.2f}%', ha='center', color='black')

    ax2 = ax.twinx()
    ax2.plot(metric_names, optimal_functional_values, color='orange', marker='o')
    ax2.set_ylabel('Functional Space Value')
    ax2.set_ylim(min(optimal_functional_values) - 5, max(optimal_functional_values) + 5)

    plt.title('Optimal Functional Value and Metric Values')
    plt.show()

    
def test_optimal_functional_space_gm(optimal_functional_space, testing_ground_truth, testing_pitch, testing_yaw, sampling_freq, similarity_metrics):
    # Using unseen testing data only
    # Compute gm scores using the new optimal_functional_space
    gm_scores = gm_algorithm(testing_pitch, testing_yaw, functional_space=optimal_functional_space)
    
    similarity_scores = {}
    
    # Loop for each metric
    for similarity_metric in similarity_metrics:
        # Use similarity metric to compare gm scores and testing_ground_truth dataset
        similarity_score = compute_similarity_metric(testing_ground_truth, gm_scores, similarity_metric)
        similarity_scores[similarity_metric] = similarity_score
    
    # Return the evaluation scores as a dictionary of similarity metrics
    return similarity_scores


def plot_similarity_metrics(similarity_metrics):
    # Convert values to percentage
    similarity_metrics = {metric: value * 100 for metric, value in similarity_metrics.items()}
    
    # Create a larger figure
    plt.figure(figsize=(10, 6))
    
    # Create bar plot
    plt.bar(range(len(similarity_metrics)), list(similarity_metrics.values()), align='center')
    
    # Add value labels on top of each bar
    for i, (metric, value) in enumerate(similarity_metrics.items()):
        plt.text(i, value + 1, f'{value:.2f}%', ha='center')
    
    # Customize plot
    plt.xticks(range(len(similarity_metrics)), list(similarity_metrics.keys()))
    plt.ylim([0, 110])  # Extend the y-axis to have room for annotations
    plt.ylabel('Percentage')
    plt.title('Similarity Metrics')
    
    # Display plot
    plt.show()


def get_prediction(data, threshold):
    """
    Computes the prediction array of 0s and 1s based on a threshold.

    Args:
        data: Numpy array of values.
        threshold: Threshold value for dichotomization.

    Returns:
        Numpy array of predictions (0s and 1s).
    """
    # Convert data to a numpy array
    data = np.array(data)

    # Compute the predictions based on the threshold
    predictions = np.where(data <= threshold, 0, 1)

    return predictions


def find_optimal_threshold(ground_truth, activity_counts):
    # Bailey and Lang, 2013
    conventional_threshold_unilateral = 2 
    
    # Convert to Numpy arrays
    ground_truth = np.array(ground_truth)
    activity_counts = np.array(activity_counts)
    
    # Define the thresholds you want to investigate
    thresholds = np.linspace(0, np.max(activity_counts), num=100000)

    # Calculate false positive rate (FPR) and true positive rate (TPR) for different thresholds
    fpr, tpr = calculate_tpr_fpr(ground_truth, activity_counts, thresholds)
    
    # Calculate the AUC (Area Under the ROC Curve)
    auc = roc_auc_score(ground_truth, activity_counts)

    # Calculate the Youden Index (Youden's J) for each threshold
    youden_index = tpr - fpr

    # Find the index of the threshold that maximizes the Youden Index
    optimal_threshold_index = np.argmax(youden_index)

    # Retrieve the optimal threshold
    optimal_threshold = thresholds[optimal_threshold_index]
    
    # Print the optimal threshold, conventional threshold, and AUC
    print(f"AUC: {auc:.2f}")
    print(f"Optimal Threshold: {optimal_threshold:.2f}")
    print(f"Conventional Threshold: {conventional_threshold_unilateral:.2f}")

    # Check if AUC is clinically useful
    if auc >= 0.75:
        print("AUC is clinically useful (≥0.75)")
    else:
        print("AUC is not clinically useful (<0.75)")

    return optimal_threshold


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


def k_fold_cross_validation(X, y, k=5, random_state=42, optimal=True):
    conventional_threshold_unilateral = 2
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    
    sensitivity_scores = []
    specificity_scores = []
    accuracy_scores = []
    ppv_scores = []
    npv_scores = []
    f1_scores = []
    youden_index_scores = []
    fpr_scores = []
    fnr_scores = []
    optimal_thresholds = []  

    for idx, (train_index, test_index) in enumerate(kf.split(X), 1):
        print(f"Iteration {idx}/{k}")
        
        X_train, X_eval = X[train_index], X[test_index]
        y_train, y_eval = y[train_index], y[test_index]
        
        if optimal:
            # Train your model and find the optimal threshold using X_train and y_train
            optimal_threshold = find_optimal_threshold(y_train, X_train)
        else: 
            optimal_threshold = conventional_threshold_unilateral
            print('Using conventional threshold')
        # Use the optimal threshold to get predictions by dichotomizing X_eval 
        pred_opt_threshold = get_prediction(X_eval, optimal_threshold)

        # Compute evaluation metrics for this iteration comparing the predictions and the y_eval_LW  
        eval_metrics = get_evaluation_metrics(y_eval, pred_opt_threshold)
            
        # Store the performance metrics for this iteration
        sensitivity_scores.append(eval_metrics['Sensitivity'])
        specificity_scores.append(eval_metrics['Specificity'])
        accuracy_scores.append(eval_metrics['Accuracy'])
        ppv_scores.append(eval_metrics['PPV'])
        npv_scores.append(eval_metrics['NPV'])
        f1_scores.append(eval_metrics['F1 Score'])
        youden_index_scores.append(eval_metrics['Youden Index'])
        fpr_scores.append(eval_metrics['False Positive Rate'])
        fnr_scores.append(eval_metrics['False Negative Rate'])

        # Store the optimal threshold for this iteration
        optimal_thresholds.append(optimal_threshold)

    # Compute the average evaluation metrics across the splits 
    avg_sensitivity = np.mean(sensitivity_scores)
    avg_specificity = np.mean(specificity_scores)
    avg_accuracy = np.mean(accuracy_scores)
    avg_ppv = np.mean(ppv_scores)
    avg_npv = np.mean(npv_scores)
    avg_f1_score = np.mean(f1_scores)
    avg_youden_index = np.mean(youden_index_scores)
    avg_fpr = np.mean(fpr_scores)
    avg_fnr = np.mean(fnr_scores)
    
    avg_eval_metrics = {
        'Sensitivity': avg_sensitivity,
        'Specificity': avg_specificity,
        'Accuracy': avg_accuracy,
        'PPV': avg_ppv,
        'NPV': avg_npv,
        'F1 Score': avg_f1_score,
        'Youden Index': avg_youden_index,
        'False Positive Rate': avg_fpr,
        'False Negative Rate': avg_fnr,
    }

    # Compute the average optimal threshold over all iterations
    avg_optimal_threshold = np.mean(optimal_thresholds)
    avg_optimal_threshold = round(avg_optimal_threshold, 2)
    
    return avg_eval_metrics, avg_optimal_threshold


def k_fold_cross_validation_bilateral(X_LW, X_RW, y_LW, y_RW, opt_threshold_LW, opt_threshold_RW,
                                      k=5, random_state=42, optimal=True):
    
    conventional_threshold_bilateral = 0
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    
    sensitivity_scores = []
    specificity_scores = []
    accuracy_scores = []
    ppv_scores = []
    npv_scores = []
    f1_scores = []
    youden_index_scores = []
    fpr_scores = []
    fnr_scores = []
    
    # Split LW and RW datasets separately 
    for idx, (train_index, test_index) in enumerate(kf.split(X_LW), 1):
        print(f"Iteration {idx}/{k}")
        
        X_train_LW, X_eval_LW = X_LW[train_index], X_LW[test_index]
        y_train_LW, y_eval_LW = y_LW[train_index], y_LW[test_index]
        
        X_train_RW, X_eval_RW = X_RW[train_index], X_RW[test_index]
        y_train_RW, y_eval_RW = y_RW[train_index], y_RW[test_index]
        
        # Get the bilateral ground truth mask  
        mask_bilateral_eval = get_mask_bilateral(y_eval_LW, y_eval_RW)
            
        if optimal:
            # Compute predictions using optimal threshold for bilateral usage
            pred_bilateral = get_prediction_bilateral(X_eval_LW, opt_threshold_LW,
                                                      X_eval_RW, opt_threshold_RW)
        else: 
            # Compute predictions using conventional threshold for bilateral usage
            pred_bilateral = get_prediction_bilateral(X_eval_LW, conventional_threshold_bilateral,
                                                      X_eval_RW, conventional_threshold_bilateral)
            
        # Get the evaluation metrics
        eval_metrics = get_evaluation_metrics(mask_bilateral_eval, pred_bilateral)
            
        # Store the performance metrics for this iteration
        sensitivity_scores.append(eval_metrics['Sensitivity'])
        specificity_scores.append(eval_metrics['Specificity'])
        accuracy_scores.append(eval_metrics['Accuracy'])
        ppv_scores.append(eval_metrics['PPV'])
        npv_scores.append(eval_metrics['NPV'])
        f1_scores.append(eval_metrics['F1 Score'])
        youden_index_scores.append(eval_metrics['Youden Index'])
        fpr_scores.append(eval_metrics['False Positive Rate'])
        fnr_scores.append(eval_metrics['False Negative Rate'])

    # Compute the average evaluation metrics across the splits 
    avg_sensitivity = np.mean(sensitivity_scores)
    avg_specificity = np.mean(specificity_scores)
    avg_accuracy = np.mean(accuracy_scores)
    avg_ppv = np.mean(ppv_scores)
    avg_npv = np.mean(npv_scores)
    avg_f1_score = np.mean(f1_scores)
    avg_youden_index = np.mean(youden_index_scores)
    avg_fpr = np.mean(fpr_scores)
    avg_fnr = np.mean(fnr_scores)
    
    avg_eval_metrics = {
        'Sensitivity': avg_sensitivity,
        'Specificity': avg_specificity,
        'Accuracy': avg_accuracy,
        'PPV': avg_ppv,
        'NPV': avg_npv,
        'F1 Score': avg_f1_score,
        'Youden Index': avg_youden_index,
        'False Positive Rate': avg_fpr,
        'False Negative Rate': avg_fnr,
        'False Negative Rate': avg_fnr
    }

    return avg_eval_metrics