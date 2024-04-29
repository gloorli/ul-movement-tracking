import numpy as np
from scipy.signal import resample
import matplotlib.pyplot as plt
from utilities import remove_extra_elements
from sklearn.model_selection import KFold
from gm_function import *
from individual_analysis_ac_functions import *
from utilities import *


def get_youden_index(gm_score, GT_mask_2Hz):
    # Calculate true positive (TP), false positive (FP), true negative (TN), and false negative (FN) counts
    TP = np.sum((gm_score == 1) & (GT_mask_2Hz == 1))
    FP = np.sum((gm_score == 1) & (GT_mask_2Hz == 0))
    TN = np.sum((gm_score == 0) & (GT_mask_2Hz == 0))
    FN = np.sum((gm_score == 0) & (GT_mask_2Hz == 1))

    # Calculate sensitivity and specificity
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)

    # Calculate the Youden Index and store it
    youden_index = sensitivity + specificity - 1
    
    return youden_index


def test_fs_values(pitch_mad, yaw_mad, GT_mask_2Hz, functional_space_array):
    gm_per_fs = []  # Initialize an empty list to store gm scores for each functional space
    
    for functional_space in functional_space_array:
        # Compute a gm score array for each possible angle using the gm_algorithm function
        gm = gm_algorithm(pitch_mad, yaw_mad, functional_space)
        
        # Ensure gm and mask have the same size 
        gm, GT_mask_2Hz = remove_extra_elements(gm, GT_mask_2Hz)
        
        # Append the computed gm score array to the list
        gm_per_fs.append(gm)  
    
    best_youden_index = -1
    optimal_fs = None
    for i, gm_score in enumerate(gm_per_fs):
        # Compute the Youden Index metric between gm score array and GT_mask_2Hz 
        youden_index = get_youden_index(gm_score, GT_mask_2Hz)
        if youden_index > best_youden_index:
            best_youden_index = youden_index
            optimal_fs = functional_space_array[i]

    return optimal_fs


def optimal_fs_computation(pitch_mad_50Hz, yaw_mad_50Hz, GT_mask_50Hz, k=5, random_state=42, optimal = True): 
    
    conventional_fs = 30 # degrees
    sensitivity_scores = []
    specificity_scores = []
    accuracy_scores = []
    ppv_scores = []
    npv_scores = []

    # Set of angles to test 
    functional_space_array = list(range(5, 91, 2))
    
    # Ensure datasets have the same size 
    pitch_mad_50Hz, GT_mask_50Hz = remove_extra_elements(pitch_mad_50Hz, GT_mask_50Hz)
    yaw_mad_50Hz, GT_mask_50Hz = remove_extra_elements(yaw_mad_50Hz, GT_mask_50Hz)
    
    # Define the X and y values that need a split 
    X1 = pitch_mad_50Hz
    X2 = yaw_mad_50Hz 
    y = GT_mask_50Hz
    
    # Split datasets into training and testing using k-fold 
    kf = KFold(n_splits=k, random_state=random_state, shuffle=True)
    
    optimal_thresholds = []
    eval_metrics = []
    
    for train_index, test_index in kf.split(X1):
        # For training dataset 
        X_train1, X_test1 = X1[train_index], X1[test_index]
        X_train2, X_test2 = X2[train_index], X2[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Downsample GT mask from 50 Hz to 2 Hz 
        y_train_2Hz = resample_mask(y_train, 50.0, 2.0)
        
        # Get optimal fs by finding the angle giving the gm score array having the highest Youden Index score when compared to the GT 
        if optimal:
            optimal_fs = test_fs_values(X_train1, X_train2, y_train_2Hz, functional_space_array)
        else:
            optimal_fs = conventional_fs
        
        # For testing dataset 
        # Downsample GT mask from 50 Hz to 2 Hz 
        y_test_2Hz = resample_mask(y_test, 50.0, 2.0)
        
        # Compute gm scores eval using angles test and optimal fs found with training data 
        gm_eval = gm_algorithm(X_test1, X_test2, functional_space = optimal_fs)
        
        # Ensure datasets have the same size 
        gm_eval, y_test_2Hz = remove_extra_elements(gm_eval, y_test_2Hz)

        # Compute eval metrics between GT test and the gm scores eval 
        eval_metrics = get_evaluation_metrics(y_test_2Hz, gm_eval)
        
       # Store the performance metrics for this iteration
        sensitivity_scores.append(eval_metrics['Sensitivity'])
        specificity_scores.append(eval_metrics['Specificity'])
        accuracy_scores.append(eval_metrics['Accuracy'])
        ppv_scores.append(eval_metrics['PPV'])
        npv_scores.append(eval_metrics['NPV'])
        
    # Compute the average evaluation metrics across the splits 
    avg_sensitivity = np.mean(sensitivity_scores)
    avg_specificity = np.mean(specificity_scores)
    avg_accuracy = np.mean(accuracy_scores)
    avg_ppv = np.mean(ppv_scores)
    avg_npv = np.mean(npv_scores)

    # Compute the average optimal fs found across the splits 
    avg_optimal_fs = np.mean(optimal_fs)
    
    avg_eval_metrics = {
        'Sensitivity': avg_sensitivity,
        'Specificity': avg_specificity,
        'Accuracy': avg_accuracy,
        'PPV': avg_ppv,
        'NPV': avg_npv,
    }

    return avg_eval_metrics, avg_optimal_fs


def optimal_fs_computation_bilateral(pitch_mad_ndh, yaw_mad_ndh, GT_mask_50Hz_ndh,
                                     pitch_mad_dh, yaw_mad_dh, GT_mask_50Hz_dh,
                                     optimal_fs_ndh, optimal_fs_dh, k=5, random_state=42, optimal=True): 
    
    conventional_fs = 30  # degrees
    sensitivity_scores = []
    specificity_scores = []
    accuracy_scores = []
    ppv_scores = []
    npv_scores = []

    # Set of angles to test 
    functional_space_array = list(range(5, 91, 2))
    
    # For ndh
    pitch_mad_ndh, GT_mask_50Hz_ndh = remove_extra_elements(pitch_mad_ndh, GT_mask_50Hz_ndh)
    yaw_mad_ndh, GT_mask_50Hz_ndh = remove_extra_elements(yaw_mad_ndh, GT_mask_50Hz_ndh)
    
    # For dh
    pitch_mad_dh, GT_mask_50Hz_dh = remove_extra_elements(pitch_mad_dh, GT_mask_50Hz_dh)
    yaw_mad_dh, GT_mask_50Hz_dh = remove_extra_elements(yaw_mad_dh, GT_mask_50Hz_dh)

    # Define the X and y values that need a split 
    # For ndh
    X1_ndh = pitch_mad_ndh
    X2_ndh = yaw_mad_ndh 
    y_ndh = GT_mask_50Hz_ndh
    # For dh
    X1_dh = pitch_mad_dh
    X2_dh = yaw_mad_dh 
    y_dh = GT_mask_50Hz_dh
    
    # Split datasets into training and testing using k-fold for ndh
    kf_ndh = KFold(n_splits=k, random_state=random_state, shuffle=True)
    # Split datasets into training and testing using k-fold for dh
    kf_dh = KFold(n_splits=k, random_state=random_state, shuffle=True)

    optimal_thresholds = []
    eval_metrics = []
    
    for (train_index_ndh, test_index_ndh), (train_index_dh, test_index_dh) in zip(kf_ndh.split(X1_ndh), kf_dh.split(X1_dh)):
        # Split the datasets for ndh
        X_train1_ndh, X_test1_ndh = X1_ndh[train_index_ndh], X1_ndh[test_index_ndh]
        X_train2_ndh, X_test2_ndh = X2_ndh[train_index_ndh], X2_ndh[test_index_ndh]
        y_train_ndh, y_test_ndh = y_ndh[train_index_ndh], y_ndh[test_index_ndh]
        
        # Split the datasets for dh
        X_train1_dh, X_test1_dh = X1_dh[train_index_dh], X1_dh[test_index_dh]
        X_train2_dh, X_test2_dh = X2_dh[train_index_dh], X2_dh[test_index_dh]
        y_train_dh, y_test_dh = y_dh[train_index_dh], y_dh[test_index_dh]
        
        # Downsample GT mask from 50 Hz to 2 Hz for ndh and for dh
        y_test_2Hz_ndh = resample_mask(y_test_ndh, 50.0, 2.0)
        y_test_2Hz_dh = resample_mask(y_test_dh, 50.0, 2.0)
        
        # Get the bilateral ground truth mask used for evaluation
        mask_bilateral_eval = get_mask_bilateral(y_test_2Hz_ndh, y_test_2Hz_dh)
        
        # Select the fs we want to evaluate
        if optimal:
            fs_ndh = optimal_fs_ndh
            fs_dh = optimal_fs_dh
        else:
            fs_ndh = conventional_fs
            fs_dh = conventional_fs
        
        # Compute gm scores eval using both ndh and dh 
        gm_eval_ndh = gm_algorithm(X_test1_ndh, X_test2_ndh, functional_space=fs_ndh)
        gm_eval_dh = gm_algorithm(X_test1_dh, X_test2_dh, functional_space=fs_dh)
        
        # Compute the bilateral GM scores using AND logic between ndh and dh gm predictions
        gm_eval_bilateral = get_mask_bilateral(gm_eval_ndh, gm_eval_dh)
        
        # Ensure datasets have the same size 
        gm_eval_bilateral, mask_bilateral_eval = remove_extra_elements(gm_eval_bilateral, mask_bilateral_eval)

        # Compute eval metrics between GT test and the gm scores eval 
        eval_metrics = get_evaluation_metrics(mask_bilateral_eval, gm_eval_bilateral)
        
        # Store the performance metrics for this iteration
        sensitivity_scores.append(eval_metrics['Sensitivity'])
        specificity_scores.append(eval_metrics['Specificity'])
        accuracy_scores.append(eval_metrics['Accuracy'])
        ppv_scores.append(eval_metrics['PPV'])
        npv_scores.append(eval_metrics['NPV'])
        
    # Compute the average evaluation metrics across the splits 
    avg_sensitivity = np.mean(sensitivity_scores)
    avg_specificity = np.mean(specificity_scores)
    avg_accuracy = np.mean(accuracy_scores)
    avg_ppv = np.mean(ppv_scores)
    avg_npv = np.mean(npv_scores)
    
    avg_eval_metrics = {
        'Sensitivity': avg_sensitivity,
        'Specificity': avg_specificity,
        'Accuracy': avg_accuracy,
        'PPV': avg_ppv,
        'NPV': avg_npv,
    }

    return avg_eval_metrics


def save_gm_arrays_as_csv(pitch_mad_ndh, yaw_mad_ndh, pitch_mad_dh, yaw_mad_dh, GT_mask_50Hz_ndh, GT_mask_50Hz_dh, folder):
    """
    Save six arrays as CSV file with headers.

    Args:
        pitch_mad_ndh (numpy.ndarray): NumPy array containing pitch_mad values for the left bronchus.
        yaw_mad_ndh (numpy.ndarray): NumPy array containing yaw_mad values for the left bronchus.
        pitch_mad_dh (numpy.ndarray): NumPy array containing pitch_mad values for the right bronchus.
        yaw_mad_dh (numpy.ndarray): NumPy array containing yaw_mad values for the right bronchus.
        GT_mask_50Hz_ndh (numpy.ndarray): NumPy array containing GT_mask_50Hz values for the left bronchus.
        GT_mask_50Hz_dh (numpy.ndarray): NumPy array containing GT_mask_50Hz values for the right bronchus.
        folder (str): Folder path to save the CSV file.

    Returns:
        None.
    """
    # Convert the arrays to DataFrames with appropriate headers
    ndh_df = pd.DataFrame({'Pitch MAD NDH': pitch_mad_ndh, 'Yaw MAD NDH': yaw_mad_ndh, 'GT Mask 50Hz NDH': GT_mask_50Hz_ndh})
    dh_df = pd.DataFrame({'Pitch MAD DH': pitch_mad_dh, 'Yaw MAD DH': yaw_mad_dh, 'GT Mask 50Hz DH': GT_mask_50Hz_dh})

    # Specify the output CSV file name
    output_filename = 'gm_datasets.csv'

    # Construct the full file path for the output CSV file
    output_path = os.path.join(folder, output_filename)

    # Ensure the folder exists, otherwise raise an error
    if not os.path.exists(folder):
        raise ValueError(f"The folder '{folder}' does not exist.")

    # Save the arrays as a single CSV file
    combined_df = pd.concat([ndh_df, dh_df], axis=1)
    combined_df.to_csv(output_path, index=False)

    # Print the path where the CSV file was saved
    if os.path.exists(output_path):
        print(f"CSV file saved successfully.")
        print(f"CSV saved at: {output_path}")
    else:
        print(f"Failed to save CSV file.")
