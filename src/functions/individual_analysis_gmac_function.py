from utilities import *
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, roc_auc_score
from tqdm import tqdm

from GMAC import get_prediction_gmac

def AUC_analisys(ground_truth, pred):
    # Calculate the AUC (Area Under the ROC Curve)
    auc = roc_auc_score(ground_truth, pred)
    print(f"AUC: {auc}")
    # Check if AUC is clinically useful
    if auc >= 0.75:
        print("AUC is clinically useful (≥0.75) according to [Fan et al., 2006]")
    else:
        print("AUC is not clinically useful (<0.75) according to [Fan et al., 2006]")
    return auc

def accuracy_analisys(accuracy: float):
    # Check if accuracy is clinically useful
    print(f"Accuracy: {accuracy*100}%")
    if accuracy >= 0.9:
        print("Accuracy is clinically useful (≥90%) according to [Lang et al., 2020]")
    else:
        print("Accuracy is not clinically useful (<90%) according to [Lang et al., 2020]")

def calculate_min_max_std(X):
    """
    Calculate the minimum, maximum and standard deviation of each feature in X.

    Args:
        X: np.array of features

    Returns:
        Numpy array of minimum, maximum and standard deviation of each feature in X.
    """
    min_max_std = {
        'min': np.min(X, axis=0),
        'max': np.max(X, axis=0),
        'std': np.std(X, axis=0)
    }

    return min_max_std

def get_threshold_range(counts, elevations):
    """
    Calculates the threshold and angle ranges based on the given maximum counts and elevations.

    Parameters:
    counts (array-like): An array-like object containing the counts.
    elevations (array-like): An array-like object containing the elevations.

    Returns:
    tuple: A tuple containing the threshold range and angle range.

    """
    thres_range = np.arange(0, int(np.max(counts)) + 1)
    angle_range = np.arange(0, int(np.max(np.abs(elevations))) + 1)
    return thres_range, angle_range

def get_classification_metrics(y_true, y_pred):
    """
    Calculate classification metrics based on the true labels and predicted labels.

    Parameters:
    - y_true (array-like): True labels.
    - y_pred (array-like): Predicted labels.

    Returns:
    - accuracy (float): Accuracy of the classification.
    - sensitivity (float): Sensitivity (True Positive Rate) of the classification.
    - specificity (float): Specificity (True Negative Rate) of the classification.
    - youden_index (float): Youden's Index of the classification.
    - F_1_score (float): F1 Score of the classification.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    youden_index = sensitivity + specificity - 1
    F_1_score = 2 * tp / (2 * tp + fp + fn)
    return accuracy, sensitivity, specificity, youden_index, F_1_score

def k_fold_cross_validation_gmac(X, y, k=5, random_state=42, optimal='No'):
    """
    Perform k-fold cross-validation on the GMAC algorithm thresholds.

    Args:
        X: 2D-Numpy array of counts and pitch
        y: Numpy array of GT labels
        k: Number of splits in k-fold cross-validation
        random_state: Random seed
        optimal: String indicating whether to find the optimal thresholds

    Returns:
        Average optimal evaluation metrics and Tuple of average optimal (count threshold, pitch threshold)
    """
    
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    
    results_test = {
        'Accuracy': [],
        'Sensitivity': [],
        'Specificity': [],
        'Youden_index': [],
        'Optimal_thres': [],
        'Optimal_angle': []
    }
    results_test = pd.DataFrame(results_test)

    results_train = {
        'Accuracy': [],
        'Sensitivity': [],
        'Specificity': [],
        'Youden_index': [],
    }
    results_train = pd.DataFrame(results_train)

    # Ensure datasets have the same size 
    X, y = remove_extra_elements(X, y)

    for idx, (train_index, test_index) in enumerate(kf.split(X), 1):
        print(f"Iteration {idx}/{k}")
        
        X_train, X_eval = X[train_index, :], X[test_index, :]
        y_train, y_eval = y[train_index], y[test_index]
        
        if optimal=='Yes_Subash':
            # Train your model and find the optimal threshold using X_train and y_train
            counts=X_train[:, 0]
            pitch=X_train[:, 1]

            # Define the thresholds you want to investigate
            thres_range, angle_range = get_threshold_range(counts, pitch)

            #vars to store values of each investigated threshold combination
            thres_angle_range = []
            results_temp = {
                'Accuracy': [],
                'Sensitivity': [],
                'Specificity': [], 
                'Youden_index': []
            }
            results_temp = pd.DataFrame(results_temp)

            # Loop through all possible threshold combinations
            for thres in thres_range:
                for angle in angle_range:
                    y_train_pred = get_prediction_gmac(counts, pitch, count_threshold=thres, functional_space=angle, decision_mode='Subash')
                    y_train_pred = y_train_pred.astype(int)

                    accuracy, sensitivity, specificity, youden_index, _ = get_classification_metrics(y_train, y_train_pred)

                    results_temp.loc[len(results_temp)] = [accuracy*100, sensitivity*100, specificity*100, youden_index]
                    thres_angle_range.append([thres, angle])

            # Find the index of the thresholds that maximizes the Youden Index
            max_index = np.argmax(results_temp['Youden_index'])
            # Retrieve the optimal thresholds
            optimal_thres = thres_angle_range[max_index][0]
            optimal_angle = thres_angle_range[max_index][1]
            results_train.loc[len(results_train)] = results_temp.iloc[max_index]
            print('Using optimized Subash GMAC thresholds')

        elif optimal=='Yes_Linus':
            # Train your model and find the optimal threshold using X_train and y_train
            counts=X_train[:, 0]
            pitch=X_train[:, 1]

            # Define the thresholds you want to investigate
            thres_range, angle_range = get_threshold_range(counts, pitch)

            #vars to store values of each investigated threshold combination
            thres_angle_range = []
            results_temp = {
                'Accuracy': [],
                'Sensitivity': [],
                'Specificity': [], 
                'Youden_index': []
            }
            results_temp = pd.DataFrame(results_temp)

            # Loop through all possible threshold combinations
            for thres in thres_range:
                for angle in angle_range:
                    y_train_pred = get_prediction_gmac(counts, pitch, count_threshold=thres, functional_space=angle, decision_mode='Linus')
                    y_train_pred = y_train_pred.astype(int)

                    accuracy, sensitivity, specificity, youden_index, _ = get_classification_metrics(y_train, y_train_pred)

                    results_temp.loc[len(results_temp)] = [accuracy*100, sensitivity*100, specificity*100, youden_index]
                    thres_angle_range.append([thres, angle])

            # Find the index of the thresholds that maximizes the Youden Index
            max_index = np.argmax(results_temp['Youden_index'])
            # Retrieve the optimal thresholds
            optimal_thres = thres_angle_range[max_index][0]
            optimal_angle = thres_angle_range[max_index][1]
            results_train.loc[len(results_train)] = results_temp.iloc[max_index]
            print('Using optimized Linus GMAC thresholds')
        
        elif optimal=='No': 
            optimal_thres = 0
            optimal_angle = 30
            print('Using conventional Subash GMAC thresholds')

        else:
            raise ValueError("Invalid value for 'optimal'. Please use 'Yes_Subash', 'Yes_Linus' or 'No'.")

        # Use the optimal thresholds to get predictions by dichotomizing X_eval 
        y_test_pred = get_prediction_gmac(counts=X_eval[:, 0], pitch=X_eval[:, 1], count_threshold=optimal_thres, functional_space=optimal_angle, decision_mode='Subash')
        y_test_pred = y_test_pred.astype(int)

        # Compute evaluation metrics for this iteration comparing the predictions and the y_eval
        accuracy, sensitivity, specificity, youden_index, _ = get_classification_metrics(y_eval, y_test_pred)

        #store metrics and thresholds of the current iteration
        results_test.loc[len(results_test)] = [accuracy*100, sensitivity*100, specificity*100, youden_index, optimal_thres, optimal_angle]

        # Print the optimal thresholds
        print(f"Optimal Count Threshold: {optimal_thres:.2f}")
        print(f"Optimal Pitch Threshold: {optimal_angle:.2f}")

    # Compute the average evaluation metrics across the splits 
    avg_sensitivity = np.mean(results_test['Sensitivity'])
    avg_specificity = np.mean(results_test['Specificity'])
    avg_accuracy = np.mean(results_test['Accuracy'])
    avg_J = np.mean(results_test['Youden_index'])
    
    avg_eval_metrics = {
        'Sensitivity': avg_sensitivity,
        'Specificity': avg_specificity,
        'Accuracy': avg_accuracy,
        'Youden_Index': avg_J,
    }

    # Compute the average optimal thresholds over all iterations
    avg_optimal_count_threshold = np.mean(results_test['Optimal_thres'])
    avg_optimal_pitch_threshold = np.mean(results_test['Optimal_angle'])
    avg_optimal_thresholds = (round(avg_optimal_count_threshold, 2), round(avg_optimal_pitch_threshold, 2))

    count_min_max_std = calculate_min_max_std(results_test['Optimal_thres'])
    pitch_min_max_std = calculate_min_max_std(results_test['Optimal_angle'])


    return avg_eval_metrics, avg_optimal_thresholds, (count_min_max_std, pitch_min_max_std)

def individual_GMAC_thresholds(X, y):
    """
    Perform nested grid search on the GMAC algorithm thresholds.

    Args:
        X: 2D-Numpy array of counts and pitch
        y: Numpy array of GT labels

    Returns:
        Tuple of individual thresholds (count threshold, pitch threshold)
    """

    # Ensure datasets have the same size 
    X, y = remove_extra_elements(X, y)

    # Train your model and find the optimal threshold using X_train and y_train
    counts=X[:, 0]
    pitch=X[:, 1]

    # Define the thresholds you want to investigate
    thres_range, angle_range = get_threshold_range(counts, pitch)

    #vars to store values of each investigated threshold combination
    thres_angle_range = []
    results_temp = {
        'Accuracy': [],
        'Sensitivity': [],
        'Specificity': [], 
        'Youden_index': []
    }
    results_temp = pd.DataFrame(results_temp)

    # Loop through all possible threshold combinations
    for thres in thres_range:
        for angle in angle_range:
            y_train_pred = get_prediction_gmac(counts, pitch, count_threshold=thres, functional_space=angle, decision_mode='Subash')
            y_train_pred = y_train_pred.astype(int)

            accuracy, sensitivity, specificity, youden_index, _ = get_classification_metrics(y, y_train_pred)

            results_temp.loc[len(results_temp)] = [accuracy*100, sensitivity*100, specificity*100, youden_index]
            thres_angle_range.append([thres, angle])

    # Find the index of the thresholds that maximizes the Youden Index
    max_index = np.argmax(results_temp['Youden_index'])
    # Retrieve the optimal thresholds
    optimal_thres = thres_angle_range[max_index][0]
    optimal_angle = thres_angle_range[max_index][1]

    # Print the optimal thresholds
    print(f"Optimal Count Threshold: {optimal_thres:.2f}")
    print(f"Optimal Pitch Threshold: {optimal_angle:.2f}")

    return [optimal_thres, optimal_angle]

def individual_GMAC_thresholds_for_all_participants(participant_ids):
    """
    Perform individual GMAC threshold analysis (nested grid search) for all participants.
    Args:
        participant_ids: List of participant IDs
    Returns:
        None
    """
    initial_path = '../data/CreateStudy'

    for participant_id in tqdm(participant_ids, desc="Processing participants"):
        participant_path = os.path.join(initial_path, participant_id)
        # Extract dataset from the participant JSON file 
        participant_data = load_participant_json(participant_id, initial_path)
        # For stroke, dominant hand = non affected hand 
        counts_for_GMAC_ndh = np.array(participant_data['counts_for_GMAC_ndh_1Hz'])
        mean_elevation_ndh = np.array(participant_data['pitch_for_GMAC_ndh_1Hz'])
        GT_mask_NDH_1Hz = np.array(participant_data['GT_mask_NDH_1Hz'])
        individual_thresholds_affected_side = individual_GMAC_thresholds(X=np.stack((counts_for_GMAC_ndh, mean_elevation_ndh), axis=0).T, y=GT_mask_NDH_1Hz)
        counts_for_GMAC_dh = np.array(participant_data['counts_for_GMAC_dh_1Hz'])
        mean_elevation_dh = np.array(participant_data['pitch_for_GMAC_dh_1Hz'])
        GT_mask_DH_1Hz = np.array(participant_data['GT_mask_DH_1Hz'])
        individual_thresholds_unaffected_side = individual_GMAC_thresholds(X=np.stack((counts_for_GMAC_dh, mean_elevation_dh), axis=0).T, y=GT_mask_DH_1Hz)
        # Store the individual thresholds for the participant
        add_attributes_to_participant(participant_data, individual_GMAC_thresholds_affected_side = individual_thresholds_affected_side, individual_GMAC_thresholds_unaffected_side = individual_thresholds_unaffected_side) 
        # Save the updated participant data
        save_to_json(participant_data, participant_path)

    print("Individual GMAC threshold analysis completed.")