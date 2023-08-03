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
    f1_scores = []
    youden_index_scores = []
    fpr_scores = []
    fnr_scores = []
    
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

    # Compute the average optimal fs found across the splits 
    avg_optimal_fs = np.mean(optimal_fs)
    
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

    return avg_eval_metrics, avg_optimal_fs


def optimal_fs_computation_bilateral(pitch_mad_LW, yaw_mad_LW, GT_mask_50Hz_LW,
                                     pitch_mad_RW, yaw_mad_RW, GT_mask_50Hz_RW,
                                     optimal_fs_LW, optimal_fs_RW, k=5, random_state=42, optimal=True): 
    
    conventional_fs = 30  # degrees
    sensitivity_scores = []
    specificity_scores = []
    accuracy_scores = []
    ppv_scores = []
    npv_scores = []
    f1_scores = []
    youden_index_scores = []
    fpr_scores = []
    fnr_scores = []
    
    # Set of angles to test 
    functional_space_array = list(range(5, 91, 2))
    
    # For LW
    pitch_mad_LW, GT_mask_50Hz_LW = remove_extra_elements(pitch_mad_LW, GT_mask_50Hz_LW)
    yaw_mad_LW, GT_mask_50Hz_LW = remove_extra_elements(yaw_mad_LW, GT_mask_50Hz_LW)
    
    # For RW
    pitch_mad_RW, GT_mask_50Hz_RW = remove_extra_elements(pitch_mad_RW, GT_mask_50Hz_RW)
    yaw_mad_RW, GT_mask_50Hz_RW = remove_extra_elements(yaw_mad_RW, GT_mask_50Hz_RW)

    # Define the X and y values that need a split 
    # For LW
    X1_LW = pitch_mad_LW
    X2_LW = yaw_mad_LW 
    y_LW = GT_mask_50Hz_LW
    # For RW
    X1_RW = pitch_mad_RW
    X2_RW = yaw_mad_RW 
    y_RW = GT_mask_50Hz_RW
    
    # Split datasets into training and testing using k-fold for LW
    kf_LW = KFold(n_splits=k, random_state=random_state, shuffle=True)
    # Split datasets into training and testing using k-fold for RW
    kf_RW = KFold(n_splits=k, random_state=random_state, shuffle=True)

    optimal_thresholds = []
    eval_metrics = []
    
    for (train_index_LW, test_index_LW), (train_index_RW, test_index_RW) in zip(kf_LW.split(X1_LW), kf_RW.split(X1_RW)):
        # Split the datasets for LW
        X_train1_LW, X_test1_LW = X1_LW[train_index_LW], X1_LW[test_index_LW]
        X_train2_LW, X_test2_LW = X2_LW[train_index_LW], X2_LW[test_index_LW]
        y_train_LW, y_test_LW = y_LW[train_index_LW], y_LW[test_index_LW]
        
        # Split the datasets for RW
        X_train1_RW, X_test1_RW = X1_RW[train_index_RW], X1_RW[test_index_RW]
        X_train2_RW, X_test2_RW = X2_RW[train_index_RW], X2_RW[test_index_RW]
        y_train_RW, y_test_RW = y_RW[train_index_RW], y_RW[test_index_RW]
        
        # Downsample GT mask from 50 Hz to 2 Hz for LW and for RW
        y_test_2Hz_LW = resample_mask(y_test_LW, 50.0, 2.0)
        y_test_2Hz_RW = resample_mask(y_test_RW, 50.0, 2.0)
        
        # Get the bilateral ground truth mask used for evaluation
        mask_bilateral_eval = get_mask_bilateral(y_test_2Hz_LW, y_test_2Hz_RW)
        
        # Select the fs we want to evaluate
        if optimal:
            fs_LW = optimal_fs_LW
            fs_RW = optimal_fs_RW
        else:
            fs_LW = conventional_fs
            fs_RW = conventional_fs
        
        # Compute gm scores eval using both LW and RW 
        gm_eval_LW = gm_algorithm(X_test1_LW, X_test2_LW, functional_space=fs_LW)
        gm_eval_RW = gm_algorithm(X_test1_RW, X_test2_RW, functional_space=fs_RW)
        
        # Compute the bilateral GM scores using AND logic between LW and RW gm predictions
        gm_eval_bilateral = get_mask_bilateral(gm_eval_LW, gm_eval_RW)
        
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
    }

    return avg_eval_metrics
