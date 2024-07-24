from utilities import *
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, roc_auc_score


def get_prediction_gmac(counts, pitch, count_threshold=0, functional_space=30, decision_mode='Subash'):
    """
    Computes the prediction array of 0s and 1s based on a a count threshold and the functional space.

    Args:
        counts: Numpy array of counts 1Hz
        pitch: Numpy array of pitch 1Hz
        count_threshold: seconds with counts above threshold are considered functional (dichotomization)
        functional_space: seconds within functional space are considered functional (dichotomization)
        decision_mode: 'Subash' or 'Linus'

    Returns:
        Numpy array of predictions (0s and 1s).
    """
    pitch_dichotomization = np.where(np.abs(pitch) < functional_space, 1, 0)# Compute the functional space dichotomization based on the original GMAC algorithm
    if decision_mode == 'Linus':
        pitch_dichotomization = np.where(pitch > -functional_space, 1, 0)# Compute the functional space dichotomization based on the Linus GMAC algorithm

    return np.where(np.logical_and(counts > count_threshold, pitch_dichotomization), 1, 0)
    #return np.where(counts > count_threshold and pitch_dichotomization, 1, 0)


def AUC_analisys(ground_truth, pred):
    # Calculate the AUC (Area Under the ROC Curve)
    auc = roc_auc_score(ground_truth, pred)
    print(f"AUC: {auc}")
    # Check if AUC is clinically useful
    if auc >= 0.75:
        print("AUC is clinically useful (≥0.75)")
    else:
        print("AUC is not clinically useful (<0.75)")
    

def k_fold_cross_validation_gmac(X, y, k=5, random_state=42, optimal=True):
    """
    Perform k-fold cross-validation on the GMAC algorithm thresholds.

    Args:
        X: 2D-Numpy array of counts and pitch
        y: Numpy array of GT labels
        k: Number of splits in k-fold cross-validation
        random_state: Random seed
        optimal: Boolean indicating whether to find the optimal thresholds

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
        
        if optimal:
            # Train your model and find the optimal threshold using X_train and y_train
            counts=X_train[:, 0]
            pitch=X_train[:, 1]

            # Define the thresholds you want to investigate
            thres_range = np.arange(0, int(np.max(counts)) + 1)
            angle_range = np.arange(0, int(np.max(np.abs(pitch))) + 1)

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

                    tn, fp, fn, tp = confusion_matrix(y_train, y_train_pred).ravel()
                    accuracy = (tp + tn) / (tp + tn + fp + fn)
                    sensitivity = tp / (tp + fn)
                    specificity = tn / (tn + fp)
                    youden_index = sensitivity + specificity - 1

                    results_temp.loc[len(results_temp)] = [accuracy*100, sensitivity*100, specificity*100, youden_index]
                    thres_angle_range.append([thres, angle])

            # Find the index of the thresholds that maximizes the Youden Index
            max_index = np.argmax(results_temp['Youden_index'])
            # Retrieve the optimal thresholds
            optimal_thres = thres_angle_range[max_index][0]
            optimal_angle = thres_angle_range[max_index][1]
            results_train.loc[len(results_train)] = results_temp.iloc[max_index]
        
        else: 
            optimal_thres = 0
            optimal_angle = 30
            print('Using conventional threshold')

        # Use the optimal thresholds to get predictions by dichotomizing X_eval 
        y_test_pred = get_prediction_gmac(counts=X_eval[:, 0], pitch=X_eval[:, 1], count_threshold=optimal_thres, functional_space=optimal_angle, decision_mode='Subash')
        y_test_pred = y_test_pred.astype(int)

        # Compute evaluation metrics for this iteration comparing the predictions and the y_eval
        tn, fp, fn, tp = confusion_matrix(y_eval, y_test_pred).ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        youden_index = sensitivity + specificity - 1

        #store metrics and thresholds of the current iteration
        results_test.loc[len(results_test)] = [accuracy*100, sensitivity*100, specificity*100, youden_index, optimal_thres, optimal_angle]

        # Print the optimal thresholds
        print(f"Optimal Count Threshold: {optimal_thres:.2f}")
        print(f"Optimal Pitch Threshold: {optimal_angle:.2f}")

        #Investigate the AUC
        #AUC_analisys(y_train, X_train) #TODO needs investigation


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

    return avg_eval_metrics, avg_optimal_thresholds