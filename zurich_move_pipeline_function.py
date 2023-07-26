import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score, jaccard_score
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d
from gm_function import *


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
    predictions = np.where(data < threshold, 0, 1)

    return predictions


def get_evaluation_metrics(ground_truth, predictions, title):
    """
    Plots the evaluation metrics for classification performance.

    Args:
        ground_truth: Numpy array of ground truth values (0s and 1s).
        predictions: Numpy array of predicted values (0s and 1s).
        title: Title for the plot (string).

    Returns:
        None (displays the plot).
    """
    # Calculate evaluation metrics
    true_positives = np.sum(np.logical_and(predictions == 1, ground_truth == 1))
    false_negatives = np.sum(np.logical_and(predictions == 0, ground_truth == 1))

    sensitivity = true_positives / (true_positives + false_negatives)
    specificity = np.sum(np.logical_and(predictions == 0, ground_truth == 0)) / np.sum(ground_truth == 0)
    accuracy = np.mean(predictions == ground_truth)

    # Calculate PPV (Positive Predictive Value) with denominator check
    positive_predictions = np.sum(predictions == 1)
    ppv = true_positives / positive_predictions if positive_predictions != 0 else 0

    # Calculate NPV (Negative Predictive Value) with denominator check
    negative_predictions = np.sum(predictions == 0)
    npv = np.sum(np.logical_and(predictions == 0, ground_truth == 0)) / negative_predictions if negative_predictions != 0 else 0

    # Convert metrics to percentages
    sensitivity *= 100
    specificity *= 100
    accuracy *= 100
    ppv *= 100
    npv *= 100

    # Prepare the metric names and values
    metric_names = ['Sensitivity', 'Specificity', 'Accuracy', 'PPV', 'NPV']
    metric_values = [sensitivity, specificity, accuracy, ppv, npv]

    # Plot the bar plot for each metric
    plt.figure(figsize=(12, 6))
    bars = plt.bar(metric_names, metric_values, width=0.5)

    # Set the figure title and y-axis label
    plt.title(title)
    plt.ylabel('Percentage')

    # Set the y-axis limit
    plt.ylim([0, 110])

    # Add the percentage values on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height,
                 f'{height:.2f}%', ha='center', va='bottom',
                 fontsize=12, fontweight='bold')  # Adjust fontsize and fontweight as desired
        
    # Adjust the x-label font properties
    plt.xticks(fontsize=12)  # Adjust fontsize and fontweight as desired

    # Show the plot
    plt.show()
    
    return {'Sensitivity': sensitivity, 'Specificity': specificity, 'Accuracy': accuracy, 'PPV': ppv, 'NPV': npv}


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


def save_metrics_dictionary_as_csv(metrics_dictionary, folder):
    """
    Saves the metrics dictionary as a CSV file in the specified folder.

    Args:
        metrics_dictionary: Dictionary with metrics data.
        folder: Folder path where the CSV file should be saved.
    """
    filename = os.path.join(folder, 'evaluation_metrics.csv')
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write the data rows
        for (vertical, horizontal), value in metrics_dictionary.items():
            writer.writerow([vertical, horizontal, value])

    print(f"The metrics dictionary has been saved as {filename}.")


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
        ('OT_RW', 'Sensitivity'): metrics_RW_OT['Sensitivity'],
        ('OT_RW', 'Specificity'): metrics_RW_OT['Specificity'],
        ('OT_RW', 'Accuracy'): metrics_RW_OT['Accuracy'],
        ('OT_RW', 'PPV'): metrics_RW_OT['PPV'],
        ('OT_RW', 'NPV'): metrics_RW_OT['NPV'],
        ('OT_bilateral', 'Sensitivity'): metrics_bilateral_OT['Sensitivity'],
        ('OT_bilateral', 'Specificity'): metrics_bilateral_OT['Specificity'],
        ('OT_bilateral', 'Accuracy'): metrics_bilateral_OT['Accuracy'],
        ('OT_bilateral', 'PPV'): metrics_bilateral_OT['PPV'],
        ('OT_bilateral', 'NPV'): metrics_bilateral_OT['NPV'],
        ('CT_LW', 'Sensitivity'): metrics_LW_CT['Sensitivity'],
        ('CT_LW', 'Specificity'): metrics_LW_CT['Specificity'],
        ('CT_LW', 'Accuracy'): metrics_LW_CT['Accuracy'],
        ('CT_LW', 'PPV'): metrics_LW_CT['PPV'],
        ('CT_LW', 'NPV'): metrics_LW_CT['NPV'],
        ('CT_RW', 'Sensitivity'): metrics_RW_CT['Sensitivity'],
        ('CT_RW', 'Specificity'): metrics_RW_CT['Specificity'],
        ('CT_RW', 'Accuracy'): metrics_RW_CT['Accuracy'],
        ('CT_RW', 'PPV'): metrics_RW_CT['PPV'],
        ('CT_RW', 'NPV'): metrics_RW_CT['NPV'],
        ('CT_bilateral', 'Sensitivity'): metrics_bilateral_CT['Sensitivity'],
        ('CT_bilateral', 'Specificity'): metrics_bilateral_CT['Specificity'],
        ('CT_bilateral', 'Accuracy'): metrics_bilateral_CT['Accuracy'],
        ('CT_bilateral', 'PPV'): metrics_bilateral_CT['PPV'],
        ('CT_bilateral', 'NPV'): metrics_bilateral_CT['NPV']
    }

    return data


def split_dataset(X, y, test_size=0.2):
    # Splitting into training and evaluation sets
    X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_eval, y_train, y_eval


def find_optimal_threshold(ground_truth, activity_counts, conventional_threshold):
    
    # Convert to Numpy arrays
    ground_truth = np.array(ground_truth)
    activity_counts = np.array(activity_counts)
    
    # Define the thresholds you want to investigate
    thresholds = np.linspace(0, np.max(activity_counts), num=100000)
    print("Thresholds tested:", thresholds)

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

    # Plot the ROC curve with the optimal threshold
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')  # Plot the diagonal line representing random guessing
    plt.scatter(fpr[optimal_threshold_index], tpr[optimal_threshold_index], color='red', label='Optimal Threshold')
    plt.scatter(fpr[conventional_threshold], tpr[conventional_threshold], color='green', label='Conventional Threshold')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()

    # Print the optimal threshold, conventional threshold, and AUC
    print(f"AUC: {auc:.2f}")
    print(f"Optimal Threshold: {optimal_threshold:.2f}")
    print(f"Conventional Threshold: {conventional_threshold:.2f}")

    # Check if AUC is clinically useful
    if auc >= 0.75:
        print("AUC is clinically useful (≥0.75)")
    else:
        print("AUC is not clinically useful (<0.75)")

    return optimal_threshold


def upsample_data(data, current_freq, desired_freq, threshold=0.5):
    # Convert data to NumPy array if needed
    if not isinstance(data, np.ndarray):
        data = np.array(data)
        data = data.flatten() 
    
    # Calculate the ratio between the current and desired frequencies
    ratio = desired_freq / current_freq

    # Create the time array corresponding to the original data
    original_time = np.arange(len(data)) / current_freq

    # Create the time array for the upsampled data
    upsampled_time = np.arange(0, len(data)-1, 1/ratio) / desired_freq

    # Create the interpolation function using cubic spline
    interpolation_func = interp1d(original_time, data, kind='cubic')

    # Perform interpolation to upsample the data
    upsampled_data = interpolation_func(upsampled_time)

    # Apply a threshold to convert the upsampled data to binary (0 or 1)
    upsampled_data = (upsampled_data > threshold).astype(int)

    return upsampled_data


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
    mask_interpolation = interp1d(original_timestamps, mask.flatten())

    # Use the interpolation function to obtain the downsampled mask values at desired timestamps
    downsampled_mask = mask_interpolation(desired_timestamps)

    # Round the interpolated values to the nearest integer (-1, 0, or 1)
    downsampled_mask = np.around(downsampled_mask).astype(int)

    return downsampled_mask


def save_optimal_threshold(file_path, left_threshold, right_threshold):
    # Create the file path
    file_path = os.path.join(file_path, 'optimal_threshold.csv')

    try:
        # Open the file in write mode
        with open(file_path, 'w', newline='') as csvfile:
            # Create a CSV writer
            csv_writer = csv.writer(csvfile)

            # Write the header row with descriptions
            csv_writer.writerow(['Side', 'Threshold'])

            # Write the left threshold as a row
            csv_writer.writerow(['Left', left_threshold])

            # Write the right threshold as a row
            csv_writer.writerow(['Right', right_threshold])

        print(f"Thresholds saved successfully at: {file_path}")
    except IOError as e:
        print(f"An error occurred while saving the thresholds: {e}")