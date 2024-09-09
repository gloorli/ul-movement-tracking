import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import spearmanr

from utilities import *
from functions.individual_analysis_gmac_function import get_prediction_gmac, AUC_analisys, accuracy_analisys, get_classification_metrics
from functions.statistics import RegressionModel
from scipy.stats import shapiro, f_oneway, ttest_rel, wilcoxon

class LOOCV_performance:
    def __init__(self, json_files):

        participant_data = extract_fields_from_json_files(json_files, ['optimal_GMAC_NDH', 'optimal_GMAC_DH', 'ARAT_score', 'FMA-UE_score', 'participant_id', 'optimal_GMAC_NDH_Linus'])

        self.PARTICIPANT_ID = participant_data['participant_id']
        self.ARAT = participant_data['ARAT_score']
        self.FMA_UE = participant_data['FMA-UE_score']

        optimal_thresholds = participant_data['optimal_GMAC_NDH']
        self.COUNT_THRESHOLD_NDH = optimal_thresholds[:,0]
        self.PITCH_THRESHOLD_NDH = optimal_thresholds[:,1]
        optimal_thresholds = participant_data['optimal_GMAC_DH']
        self.COUNT_THRESHOLD_DH = optimal_thresholds[:,0]
        self.PITCH_THRESHOLD_DH = optimal_thresholds[:,1]
        optimal_thresholds = participant_data['optimal_GMAC_NDH_Linus']
        self.COUNT_THRESHOLD_NDH_Linus = optimal_thresholds[:,0]
        self.PITCH_THRESHOLD_NDH_Linus = optimal_thresholds[:,1]

        elevation_NDH_1Hz = []
        elevation_DH_1Hz = []
        counts_NDH_1Hz = []
        counts_DH_1Hz = []
        GT_NDH_1Hz = []
        GT_DH_1Hz = []
        task_mask_ndh_1Hz = []
        task_mask_dh_1Hz = []
        for path in json_files:
            dict_1Hz = extract_fields_from_json_files([path], ['GT_mask_NDH_1Hz', 'GT_mask_DH_1Hz', 'counts_for_GMAC_ndh_1Hz', 'counts_for_GMAC_dh_1Hz', 'task_mask_for_GMAC_NDH_1Hz', 'task_mask_for_GMAC_DH_1Hz', 'pitch_for_GMAC_ndh_1Hz', 'pitch_for_GMAC_dh_1Hz'])
            task_mask_ndh_1Hz.append(dict_1Hz['task_mask_for_GMAC_NDH_1Hz'])
            task_mask_dh_1Hz.append(dict_1Hz['task_mask_for_GMAC_DH_1Hz'])
            GT_NDH_1Hz.append(dict_1Hz['GT_mask_NDH_1Hz'])
            GT_DH_1Hz.append(dict_1Hz['GT_mask_DH_1Hz'])
            counts_NDH_1Hz.append(dict_1Hz['counts_for_GMAC_ndh_1Hz'])
            counts_DH_1Hz.append(dict_1Hz['counts_for_GMAC_dh_1Hz'])
            elevation_NDH_1Hz.append(dict_1Hz['pitch_for_GMAC_ndh_1Hz'])
            elevation_DH_1Hz.append(dict_1Hz['pitch_for_GMAC_dh_1Hz'])
        task_mask = {'task_mask_ndh_1Hz': task_mask_ndh_1Hz, 'task_mask_dh_1Hz': task_mask_dh_1Hz}
        gt_functional = {'GT_mask_NDH_1Hz': GT_NDH_1Hz, 'GT_mask_DH_1Hz': GT_DH_1Hz}
        count_data = {'counts_NDH_1Hz': counts_NDH_1Hz, 'counts_DH_1Hz': counts_DH_1Hz}
        pitch_data = {'elevation_NDH_1Hz': elevation_NDH_1Hz, 'elevation_DH_1Hz': elevation_DH_1Hz}
        self.task_mask = task_mask
        self.gt_functional = gt_functional
        self.count_data = count_data
        self.pitch_data = pitch_data

    def fit_desired_model(self, model, model_type: str):
        if model_type == 'linear':
            model.fit_linear_regression()
        elif model_type == 'polynomial':
            model.fit_polynomial_regression(2)
        else:
            raise ValueError(f"Invalid model_type: {model_type}")
        return model

    def get_threshold_model(self, X_train, count_threshold_model='linear', elevation_threshold_model='polynomial', decision_mode='Subash'):
        FMA_array = np.array([participant['FMA_UE'] for participant in X_train])
        
        count_threshold_array = np.array([participant['COUNT_THRESHOLD_NDH'] for participant in X_train])
        elevation_threshold_array = np.array([participant['PITCH_THRESHOLD_NDH'] for participant in X_train])
        if decision_mode == 'Linus': #get individual optimal thresholds for Linus definition of GM rule
            count_threshold_array = np.array([participant['COUNT_THRESHOLD_NDH_Linus'] for participant in X_train])
            elevation_threshold_array = np.array([participant['PITCH_THRESHOLD_NDH_Linus'] for participant in X_train])

        regression_model_count = RegressionModel(FMA_array, count_threshold_array)
        regression_model_elevation = RegressionModel(FMA_array, elevation_threshold_array)

        return self.fit_desired_model(regression_model_count, count_threshold_model), self.fit_desired_model(regression_model_elevation, elevation_threshold_model)

    def retreive_personalized_thresholds(self, X_test, count_threshold_model, elevation_threshold_model):
        assert len(X_test) == 1, "X_test should contain only one participant"
        FMA = X_test[0]['FMA_UE']
        count_predict = count_threshold_model.predict_linear(FMA)
        elevation_predict = elevation_threshold_model.predict_polynomial(FMA, 2)

        print("Validation on participant ", X_test[0]['participant_id'])
        print("Linear Predictions of personal COUNT threshold: ", count_predict, ". Ground truth individual optimal threshold NDH: ", X_test[0]['COUNT_THRESHOLD_NDH'])
        print("Polynomial Predictions of personal ELEVATION threshold: ", elevation_predict, ". Ground truth individual optimal threshold NDH: ", X_test[0]['PITCH_THRESHOLD_NDH'])

        return count_predict, elevation_predict
    
    def retreive_mean_thresholds(self, X_train, decision_mode='Subash', side='NDH'):
        assert decision_mode != 'Linus' or side != 'DH', "Invalid combination of decision mode and side. (Linus, DH) is not yet implemented."
        count_threshold_array = np.array([participant['COUNT_THRESHOLD_NDH'] for participant in X_train])
        elevation_threshold_array = np.array([participant['PITCH_THRESHOLD_NDH'] for participant in X_train])
        if decision_mode=='Linus':
            count_threshold_array = np.array([participant['COUNT_THRESHOLD_NDH_Linus'] for participant in X_train])
            elevation_threshold_array = np.array([participant['PITCH_THRESHOLD_NDH_Linus'] for participant in X_train])
        if side == 'DH':
            count_threshold_array = np.array([participant['COUNT_THRESHOLD_DH'] for participant in X_train])
            elevation_threshold_array = np.array([participant['PITCH_THRESHOLD_DH'] for participant in X_train])

        return np.mean(count_threshold_array), np.mean(elevation_threshold_array)

    def calculate_GMAC_classification_performance(self, X_test, y_test, personalized_count_threshold, personalized_elevation_threshold, side='NDH'):
        assert len(X_test) == 1, "X_test should contain only one participant"
        assert len(y_test) == 1, "y_test should contain only one participant"
        assert side in ['NDH', 'DH'], "Invalid side: {side}"
        test_gt = y_test[0].flatten()

        counts_array = np.array(X_test[0]['counts_NDH_1Hz']) if side == 'NDH' else np.array(X_test[0]['counts_DH_1Hz'])
        elevation_array = np.array(X_test[0]['elevation_NDH_1Hz']) if side == 'NDH' else np.array(X_test[0]['elevation_DH_1Hz'])

        gmac_prediction = get_prediction_gmac(counts_array, elevation_array, count_threshold=personalized_count_threshold, functional_space=personalized_elevation_threshold, decision_mode='Subash')
        gmac_prediction = gmac_prediction.astype(int)

        accuracy, _, _, youden_index, F_1_score = get_classification_metrics(test_gt, gmac_prediction.flatten())
        auc = AUC_analisys(test_gt, gmac_prediction.flatten())
        accuracy_analisys(accuracy)

        return youden_index, accuracy, auc, F_1_score
    
    def calculate_GMAC_classification_performance_perTask(self, task_dict, count_threshold, elevation_threshold):
        accuracy_per_task = {}
        for task_of_interest, task_data in task_dict.items():
            count_for_task = np.array(task_data['count'])
            pitch_for_task = np.array(task_data['elevation'])
            gt_for_task = task_data['gt'].flatten()

            gmac_prediction = get_prediction_gmac(count_for_task, pitch_for_task, count_threshold=count_threshold, functional_space=elevation_threshold, decision_mode='Subash')
            gmac_prediction = gmac_prediction.astype(int)

            tn, fp, fn, tp = confusion_matrix(gt_for_task, gmac_prediction, labels=[1, 0]).ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)

            accuracy_per_task[task_of_interest] = accuracy

        return accuracy_per_task

    def prepare_participant_dicts(self):
        """
        Prepare a list of dictionaries, where each dictionary represents a participant and contains their relevant data.
        """
        participant_dicts = []
        for i, participant_id in enumerate(self.PARTICIPANT_ID):
            participant_dict = {
                'participant_id': participant_id,
                'ARAT': self.ARAT[i],
                'FMA_UE': self.FMA_UE[i],
                'COUNT_THRESHOLD_NDH': self.COUNT_THRESHOLD_NDH[i],
                'PITCH_THRESHOLD_NDH': self.PITCH_THRESHOLD_NDH[i],
                'COUNT_THRESHOLD_NDH_Linus': self.COUNT_THRESHOLD_NDH_Linus[i],
                'PITCH_THRESHOLD_NDH_Linus': self.PITCH_THRESHOLD_NDH_Linus[i],
                'COUNT_THRESHOLD_DH': self.COUNT_THRESHOLD_DH[i],
                'PITCH_THRESHOLD_DH': self.PITCH_THRESHOLD_DH[i],
                'counts_NDH_1Hz': self.count_data['counts_NDH_1Hz'][i],
                'elevation_NDH_1Hz': self.pitch_data['elevation_NDH_1Hz'][i],
                'task_mask_ndh_1Hz': self.task_mask['task_mask_ndh_1Hz'][i],
                'counts_DH_1Hz': self.count_data['counts_DH_1Hz'][i],
                'elevation_DH_1Hz': self.pitch_data['elevation_DH_1Hz'][i],
                'task_mask_dh_1Hz': self.task_mask['task_mask_dh_1Hz'][i],
            }
            participant_dicts.append(participant_dict)
        return participant_dicts

    def LOOCV_complete(self, perTask=True):
        """
        Performs Leave-One-Subject-Out Cross Validation (LOOCV) for classification performance evaluation.
        Args:
            perTask (bool, optional): Flag indicating whether to calculate performance metrics per task. 

        Returns:
            None
        """
        looCV = LeaveOneGroupOut()
        X = self.prepare_participant_dicts()
        y_ndh = self.gt_functional['GT_mask_NDH_1Hz']
        y_dh = self.gt_functional['GT_mask_DH_1Hz']
        group_IDs = self.PARTICIPANT_ID

        self.evaluation_FMA = []

        self.optimal_YI_list_ndh = []
        self.conventioanl_YI_list_ndh = []
        self.mean_YI_list_ndh = []
        self.optimal_YI_list_ndh_Linus = []
        self.mean_YI_list_ndh_Linus = []
        self.conventional_YI_list_dh = []
        self.mean_YI_list_dh = []

        self.optimal_accuracy_list_ndh = []
        self.conventioanl_accuracy_list_ndh = []
        self.mean_accuracy_list_ndh = []
        self.optimal_accuracy_list_ndh_Linus = []
        self.mean_accuracy_list_ndh_Linus = []
        self.conventional_accuracy_list_dh = []
        self.mean_accuracy_list_dh = []

        self.optimal_AUC_list_ndh = []
        self.conventioanl_AUC_list_ndh = []
        self.mean_AUC_list_ndh = []
        self.optimal_AUC_list_ndh_Linus = []
        self.mean_AUC_list_ndh_Linus = []
        self.conventional_AUC_list_dh = []
        self.mean_AUC_list_dh = []

        self.optimal_F1_list_ndh = []
        self.conventioanl_F1_list_ndh = []
        self.mean_F1_list_ndh = []
        self.optimal_F1_list_ndh_Linus = []
        self.mean_F1_list_ndh_Linus = []
        self.conventional_F1_list_dh = []
        self.mean_F1_list_dh = []

        self.optimal_accuracy_perTask = {}
        self.conventional_accuracy_perTask = {}
        self.mean_accuracy_perTask_dh = {}

        for train_index, test_index in looCV.split(X, y_ndh, groups=group_IDs):
            
            X_train, _ = [X[i] for i in train_index], [y_ndh[i] for i in train_index]
            X_test, y_test = [X[i] for i in test_index], [y_ndh[i] for i in test_index]
            y_test_dh = [y_dh[i] for i in test_index]

            regression_model_count_ndh, regression_model_elevation_ndh = self.get_threshold_model(X_train)
            regression_model_count_ndh_Linus, regression_model_elevation_ndh_Linus = self.get_threshold_model(X_train, decision_mode='Linus')
            
            personalized_count_threshold_ndh, personalized_elevation_threshold_ndh = self.retreive_personalized_thresholds(X_test, regression_model_count_ndh, regression_model_elevation_ndh)
            personalized_count_threshold_ndh_Linus, personalized_elevation_threshold_ndh_Linus = self.retreive_personalized_thresholds(X_test, regression_model_count_ndh_Linus, regression_model_elevation_ndh_Linus)

            youden_index_optimized_ndh, accuracy_optimized_ndh, auc_optimized_ndh, F1_optimized_ndh = self.calculate_GMAC_classification_performance(X_test, y_test, personalized_count_threshold_ndh, personalized_elevation_threshold_ndh, side='NDH')
            youden_index_conventional_ndh, accuracy_conventional_ndh, auc_conventional_ndh, F1_conventional_ndh = self.calculate_GMAC_classification_performance(X_test, y_test, 0, 30, side='NDH')
            youden_index_optimized_ndh_Linus, accuracy_optimized_ndh_Linus, auc_optimized_ndh_Linus, F1_optimized_ndh_Linus = self.calculate_GMAC_classification_performance(X_test, y_test, personalized_count_threshold_ndh_Linus, personalized_elevation_threshold_ndh_Linus, side='NDH')

            loocv_mean_count_ndh, loocv_mean_elevation_ndh = self.retreive_mean_thresholds(X_train)
            youden_index_mean_ndh, accuracy_mean_ndh, auc_mean_ndh, F1_mean_ndh = self.calculate_GMAC_classification_performance(X_test, y_test, loocv_mean_count_ndh, loocv_mean_elevation_ndh, side='NDH')
            loocv_mean_count_ndh_Linus, loocv_mean_elevation_ndh_Linus = self.retreive_mean_thresholds(X_train, decision_mode='Linus')
            youden_index_mean_ndh_Linus, accuracy_mean_ndh_Linus, auc_mean_ndh_Linus, F1_mean_ndh_Linus = self.calculate_GMAC_classification_performance(X_test, y_test, loocv_mean_count_ndh_Linus, loocv_mean_elevation_ndh_Linus, side='NDH')

            youden_index_conventional_dh, accuracy_conventional_dh, auc_conventional_dh, F1_conventional_dh = self.calculate_GMAC_classification_performance(X_test, y_test_dh, 0, 30, side='DH')
            loocv_mean_count_dh, loocv_mean_elevation_dh = self.retreive_mean_thresholds(X_train, side='DH')
            youden_index_mean_dh, accuracy_mean_dh, auc_mean_dh, F1_mean_dh = self.calculate_GMAC_classification_performance(X_test, y_test_dh, loocv_mean_count_dh, loocv_mean_elevation_dh, side='DH')

            self.evaluation_FMA.append(X_test[0]['FMA_UE'])

            self.optimal_YI_list_ndh.append(youden_index_optimized_ndh)
            self.conventioanl_YI_list_ndh.append(youden_index_conventional_ndh)
            self.mean_YI_list_ndh.append(youden_index_mean_ndh)
            self.optimal_YI_list_ndh_Linus.append(youden_index_optimized_ndh_Linus)
            self.mean_YI_list_ndh_Linus.append(youden_index_mean_ndh_Linus)
            self.conventional_YI_list_dh.append(youden_index_conventional_dh)
            self.mean_YI_list_dh.append(youden_index_mean_dh)

            self.optimal_accuracy_list_ndh.append(accuracy_optimized_ndh)
            self.conventioanl_accuracy_list_ndh.append(accuracy_conventional_ndh)
            self.mean_accuracy_list_ndh.append(accuracy_mean_ndh)
            self.optimal_accuracy_list_ndh_Linus.append(accuracy_optimized_ndh_Linus)
            self.mean_accuracy_list_ndh_Linus.append(accuracy_mean_ndh_Linus)
            self.conventional_accuracy_list_dh.append(accuracy_conventional_dh)
            self.mean_accuracy_list_dh.append(accuracy_mean_dh)

            self.optimal_AUC_list_ndh.append(auc_optimized_ndh)
            self.conventioanl_AUC_list_ndh.append(auc_conventional_ndh)
            self.mean_AUC_list_ndh.append(auc_mean_ndh)
            self.optimal_AUC_list_ndh_Linus.append(auc_optimized_ndh_Linus)
            self.mean_AUC_list_ndh_Linus.append(auc_mean_ndh_Linus)
            self.conventional_AUC_list_dh.append(auc_conventional_dh)
            self.mean_AUC_list_dh.append(auc_mean_dh)

            self.optimal_F1_list_ndh.append(F1_optimized_ndh)
            self.conventioanl_F1_list_ndh.append(F1_conventional_ndh)
            self.mean_F1_list_ndh.append(F1_mean_ndh)
            self.optimal_F1_list_ndh_Linus.append(F1_optimized_ndh_Linus)
            self.mean_F1_list_ndh_Linus.append(F1_mean_ndh_Linus)
            self.conventional_F1_list_dh.append(F1_conventional_dh)
            self.mean_F1_list_dh.append(F1_mean_dh)
            #TODO plot predicted and ground truth personalized count threshold and elevation threshold

            if perTask:
                #TODO add dominant hand
                self.LOOCV_perTask(X_test, y_test, y_test_dh, personalized_count_threshold_ndh, personalized_elevation_threshold_ndh, loocv_mean_count_dh, loocv_mean_elevation_dh)

    def LOOCV_perTask(self, X_test, y_test, y_test_dh, personalized_count_threshold_ndh, personalized_elevation_threshold_ndh, loocv_mean_count_dh, loocv_mean_elevation_dh):
        protocol_tasks = ['open_bottle_and_pour_glass', 'drink', 'fold_rags_towels', 'sort_documents', 'brooming', 'putting_on_and_off_coat', 'keyboard_typing', 'stapling', 'walking', 
                          'open_and_close_door', 'resting', 'other', 'wipe_table','light_switch']
        task_dict = {}
        task_dict_dh = {}
        for task_of_interest in protocol_tasks: #attention: hardcoded for ndh
            count_for_task = extract_all_values_with_label(X_test[0]['counts_NDH_1Hz'], X_test[0]['task_mask_ndh_1Hz'], task_of_interest)
            pitch_for_task = extract_all_values_with_label(X_test[0]['elevation_NDH_1Hz'], X_test[0]['task_mask_ndh_1Hz'], task_of_interest)
            gt_for_task = extract_all_values_with_label(y_test[0], X_test[0]['task_mask_ndh_1Hz'], task_of_interest)
            task_dict[task_of_interest] = {'count': count_for_task, 'elevation': pitch_for_task, 'gt': gt_for_task}

            count_for_task_dh = extract_all_values_with_label(X_test[0]['counts_DH_1Hz'], X_test[0]['task_mask_dh_1Hz'], task_of_interest)
            pitch_for_task_dh = extract_all_values_with_label(X_test[0]['elevation_DH_1Hz'], X_test[0]['task_mask_dh_1Hz'], task_of_interest)
            gt_for_task_dh = extract_all_values_with_label(y_test_dh[0], X_test[0]['task_mask_dh_1Hz'], task_of_interest)
            task_dict_dh[task_of_interest] = {'count': count_for_task_dh, 'elevation': pitch_for_task_dh, 'gt': gt_for_task_dh}

        accuracy_per_task_optimized = self.calculate_GMAC_classification_performance_perTask(task_dict, personalized_count_threshold_ndh, personalized_elevation_threshold_ndh)
        accuracy_per_task_conventional = self.calculate_GMAC_classification_performance_perTask(task_dict, 0, 30)
        accuracy_per_task_mean_dh = self.calculate_GMAC_classification_performance_perTask(task_dict_dh,loocv_mean_count_dh, loocv_mean_elevation_dh)

        for task, accuracy in accuracy_per_task_optimized.items():
            if task in self.optimal_accuracy_perTask:
                self.optimal_accuracy_perTask[task].append(accuracy)
            else:
                self.optimal_accuracy_perTask[task] = [accuracy]
        for task, accuracy in accuracy_per_task_conventional.items():
            if task in self.conventional_accuracy_perTask:
                self.conventional_accuracy_perTask[task].append(accuracy)
            else:
                self.conventional_accuracy_perTask[task] = [accuracy]
        for task, accuracy in accuracy_per_task_mean_dh.items():
            if task in self.mean_accuracy_perTask_dh:
                self.mean_accuracy_perTask_dh[task].append(accuracy)
            else:
                self.mean_accuracy_perTask_dh[task] = [accuracy]
    
    def check_ANOVA_ttest_Wilcoxon(self, optimal_distribution, conventional_distribution, mean_distribution, mean_distribution_dh=None, conventional_distribution_dh=None):
        """
        Check the statistical significance of the differences between the classification performance of the different GMAC thresholds applied using ANOVA, paired t-test, and Wilcoxon signed-rank test.
        Parameters:
        - optimal_distribution (array-like): The distribution of classification performance for the optimal thresholds.
        - conventional_distribution (array-like): The distribution of classification performance for the conventional thresholds.
        - mean_distribution (array-like): The distribution of classification performance for the mean count and elevation thresholds.
        - mean_distribution_dh (array-like, optional): The distribution of classification performance for the mean count and elevation thresholds for the dominant hand. Default is None.
        - conventional_distribution_dh (array-like, optional): The distribution of classification performance for the conventional thresholds for the dominant hand. Default is None.
        Note:
        - The null hypothesis for the Shapiro-Wilk test is that the data is normally distributed.
        - The null hypothesis for the one-way ANOVA is that there is no significant difference between the groups.
        - The null hypothesis for the paired t-tests is that there is no significant difference between the paired samples.
        - The null hypothesis for the Wilcoxon signed-rank tests is that there is no significant difference between the paired samples.
        """
        # Check for normality using Shapiro-Wilk test
        _, p_optimal = shapiro(optimal_distribution)
        _, p_conventional = shapiro(conventional_distribution)
        _, p_mean = shapiro(mean_distribution)
        _, p_mean_dh = shapiro(mean_distribution_dh)
        _, p_conventional_dh = shapiro(conventional_distribution_dh)
        # Perform one-way ANOVA
        _, p_value = f_oneway(optimal_distribution, conventional_distribution, mean_distribution, mean_distribution_dh, conventional_distribution_dh)

        if p_optimal > 0.05 and p_conventional > 0.05 and p_mean > 0.05 and p_mean_dh > 0.05 and p_conventional_dh > 0.05:
            print("The data is normally distributed.")
        else:
            print("The data is not normally distributed.")
        if p_value < 0.05:
            print("There is a significant ANOVA difference between the groups.")
        else:
            print("There is no significant ANOVA difference between the groups.")
        
        # Perform paired t-test
        _, p_value_optimal_conventional = ttest_rel(optimal_distribution, conventional_distribution)
        _, p_value_optimal_mean = ttest_rel(optimal_distribution, mean_distribution)
        _, p_value_conventional_mean = ttest_rel(conventional_distribution, mean_distribution)
        _, p_value_mean_conventional_dh = ttest_rel(mean_distribution_dh, conventional_distribution_dh)
        print(f"Paired t-test optimal vs conventional: {p_value_optimal_conventional}")
        print(f"Paired t-test optimal vs mean: {p_value_optimal_mean}")
        print(f"Paired t-test conventional vs mean: {p_value_conventional_mean}")
        print(f"Paired t-test mean vs conventional DH: {p_value_mean_conventional_dh}")
        
        # Perform Wilcoxon signed-rank test
        _, p_value_optimal_conventional_wilcoxon = wilcoxon(optimal_distribution, conventional_distribution)
        _, p_value_optimal_mean_wilcoxon = wilcoxon(optimal_distribution, mean_distribution)
        _, p_value_conventional_mean_wilcoxon = wilcoxon(conventional_distribution, mean_distribution)
        _, p_value_mean_conventional_dh_wilcoxon = wilcoxon(mean_distribution_dh, conventional_distribution_dh)
        print(f"Wilcoxon signed-rank test optimal vs conventional: {p_value_optimal_conventional_wilcoxon}")
        print(f"Wilcoxon signed-rank test optimal vs mean: {p_value_optimal_mean_wilcoxon}")
        print(f"Wilcoxon signed-rank test conventional vs mean: {p_value_conventional_mean_wilcoxon}")
        print(f"Wilcoxon signed-rank test mean vs conventional DH: {p_value_mean_conventional_dh_wilcoxon}")

        return p_value_optimal_conventional, p_value_optimal_mean, p_value_conventional_mean, p_value_mean_conventional_dh

    def plot_significance_brackets(self, ax, bracket_positions, p_values, bracket_heights, position="above"):
        """
        Adds significance brackets with p-values to the plot, either above or below the boxplots.
        
        Parameters:
        - ax: The axis to plot on.
        - bracket_positions: List of tuples with (start, end) x-axis positions for brackets.
        - p_values: List of p-values corresponding to each bracket.
        - bracket_heights: List of heights (y-coordinates) for the brackets.
        - position: Either 'above' or 'below' to position the brackets accordingly.
        """
        for (start, end), p_val, y in zip(bracket_positions, p_values, bracket_heights):
            x1, x2 = start, end  # x-coordinates of the brackets
            h, col = 0.02, 'k'  # Adjust height and color of the bracket
            
            # Adjust y-coordinates based on position ('above' or 'below')
            if position == "below":
                y = y - 0.05  # Shift the bracket lower

            if position == "above":
                ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.0, c=col)
            elif position == "below":
                ax.plot([x1, x1, x2, x2], [y, y-h, y-h, y], lw=1.0, c=col)
            
            # Format p-value with 3 decimal places and reduce font size
            ax.text((x1 + x2) * .5, y + h if position == "above" else y - 2*h, f"p = {p_val:.4f}", 
                    ha='center', va='bottom', color=col, fontsize=8)

    #TODO combine the following three functions into one
    def plot_LOOCV_YoudenIndex(self):
        colors = thesis_style.get_thesis_colours()
        # Get p-values from the test
        ttest_pvalue_optimal_conventional, ttest_pvalue_optimal_mean, ttest_pvalue_conventional_mean, ttest_pvalue_mean_conventional_dh = self.check_ANOVA_ttest_Wilcoxon(
            self.optimal_YI_list_ndh, self.conventioanl_YI_list_ndh, self.mean_YI_list_ndh, self.mean_YI_list_dh, self.conventional_YI_list_dh
        )
        
        mean_markers = dict(marker='D', markerfacecolor=colors['black'], markersize=5, markeredgewidth=0)
        median_markers = dict(color=colors['black_grey'])

        fig, ax = plt.subplots(figsize=(12, 6))

        # Boxplots
        box_conventional = ax.boxplot(self.conventioanl_YI_list_ndh, positions=[1], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=median_markers)
        box_mean = ax.boxplot(self.mean_YI_list_ndh, positions=[2], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=median_markers)
        box_optimal = ax.boxplot(self.optimal_YI_list_ndh, positions=[3], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=median_markers)
        box_conventional_dh = ax.boxplot(self.conventional_YI_list_dh, positions=[4], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=median_markers)
        box_mean_dh = ax.boxplot(self.mean_YI_list_dh, positions=[5], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=median_markers)

        # Set box colors
        for box in box_conventional['boxes']:
            box.set(facecolor=colors['affected'], alpha=0.7)
        for box in box_mean['boxes']:
            box.set(facecolor=colors['affected'], alpha=0.7)
        for box in box_optimal['boxes']:
            box.set(facecolor=colors['affected'], alpha=0.7)
        for box in box_conventional_dh['boxes']:
            box.set(facecolor=colors['healthy'], alpha=0.7)
        for box in box_mean_dh['boxes']:
            box.set(facecolor=colors['healthy'], alpha=0.7)

        # random classifier
        ax.axhline(y=0.0, color=colors['black_grey'], linestyle='--', label='Performance of random classifier')
        ax.add_artist(plt.legend(loc='lower right'))

        # Add colors of healthy and affected boxes to legend
        legend_colors = [colors['healthy'], colors['affected']]
        legend_labels = ['Healthy side', 'Affected side']
        legend_patches = [mpatches.Patch(color=color, label=label, alpha=0.7) for color, label in zip(legend_colors, legend_labels)]
        ax.add_artist(plt.legend(handles=legend_patches, loc='upper right'))

        # Set x-ticks and labels
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xticklabels(['Conventional', 'Mean personalized', 'Personalized', 'Conventional', 'Mean personalized'])
        ax.set_ylim(-0.1, 1)

        # Set y-label and title
        plt.ylabel('Youden Index')
        plt.title('Leave one Participant out Cross Validation')

        # Define significance bracket positions, p-values, and heights
        bracket_heights = [0.65, 0.7, 0.75, 0.7]  # Different heights for the brackets above the boxplot
        bracket_positions = [(1, 3), (2, 3), (1, 2), (4, 5)]  # (start, end) of the brackets
        p_values = [ttest_pvalue_optimal_conventional, ttest_pvalue_optimal_mean, ttest_pvalue_conventional_mean, ttest_pvalue_mean_conventional_dh]
        self.plot_significance_brackets(ax, bracket_positions, p_values, bracket_heights, position="above")

        plt.show()

    def plot_LOOCV_AUC(self):
        colors = thesis_style.get_thesis_colours()
        ttest_pvalue_optimal_conventional, ttest_pvalue_optimal_mean, ttest_pvalue_conventional_mean, ttest_pvalue_mean_conventional_dh = self.check_ANOVA_ttest_Wilcoxon(
            self.optimal_AUC_list_ndh, self.conventioanl_AUC_list_ndh, self.mean_AUC_list_ndh, self.mean_AUC_list_dh, self.conventional_AUC_list_dh
            )

        mean_markers = dict(marker='D', markerfacecolor=colors['black'], markersize=5, markeredgewidth=0)
        meadian_markers = dict(color=colors['black_grey'])

        fig, ax = plt.subplots(figsize=(12, 6))

        box_conventional = ax.boxplot(self.conventioanl_AUC_list_ndh, positions=[1], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers)
        box_mean = ax.boxplot(self.mean_AUC_list_ndh, positions=[2], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers)
        box_optimal = ax.boxplot(self.optimal_AUC_list_ndh, positions=[3], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers)
        box_conventional_dh = ax.boxplot(self.conventional_AUC_list_dh, positions=[4], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers)
        box_mean_dh = ax.boxplot(self.mean_AUC_list_dh, positions=[5], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers)

        # Set box colors
        for box in box_conventional['boxes']:
            box.set(facecolor=colors['affected'], alpha=0.7)
        for box in box_mean['boxes']:
            box.set(facecolor=colors['affected'], alpha=0.7)
        for box in box_optimal['boxes']:
            box.set(facecolor=colors['affected'], alpha=0.7)
        for box in box_conventional_dh['boxes']:
            box.set(facecolor=colors['healthy'], alpha=0.7)
        for box in box_mean_dh['boxes']:
            box.set(facecolor=colors['healthy'], alpha=0.7)

        # AUC is clinically useful (≥0.75) according to [Fan et al., 2006]
        ax.axhline(y=0.75, color=colors['pink'], linestyle='dotted', label='Clinically required performance', lw=2.0)
        # random classifier
        ax.axhline(y=0.5, color=colors['black_grey'], linestyle='--', label='Performance of random classifier')
        ax.add_artist(plt.legend(loc='upper right'))

        # Add colors of healthy and affected boxes to legend
        legend_colors = [colors['healthy'], colors['affected']]
        legend_labels = ['Healthy side', 'Affected side']
        legend_patches = [mpatches.Patch(color=color, label=label, alpha=0.7) for color, label in zip(legend_colors, legend_labels)]
        ax.add_artist(plt.legend(handles=legend_patches, loc='lower right'))

        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xticklabels(['Conventional', 'Mean personalized', 'Personalized', 'Conventional', 'Mean personalized'])
        ax.set_ylim(0.45, 1.0)

        plt.rcParams.update({'font.size': 12})
        plt.ylabel('Area Under the Receiver Operating Characteristic Curve \n(ROC AUC)')
        plt.title('Leave one Participant out Cross Validation')

        # Define significance bracket positions, p-values, and heights
        bracket_heights = [0.81, 0.9, 0.86, 0.86]  # Different heights for the brackets above the boxplot
        bracket_positions = [(1, 3), (2, 3), (1, 2), (4, 5)]  # (start, end) of the brackets
        p_values = [ttest_pvalue_optimal_conventional, ttest_pvalue_optimal_mean, ttest_pvalue_conventional_mean, ttest_pvalue_mean_conventional_dh]
        self.plot_significance_brackets(ax, bracket_positions, p_values, bracket_heights, position="above")

        plt.show()

    def plot_LOOCV_Accuracy(self):
        colors = thesis_style.get_thesis_colours()
        ttest_pvalue_optimal_conventional, ttest_pvalue_optimal_mean, ttest_pvalue_conventional_mean, ttest_pvalue_mean_conventional_dh = self.check_ANOVA_ttest_Wilcoxon(
            self.optimal_accuracy_list_ndh, self.conventioanl_accuracy_list_ndh, self.mean_accuracy_list_ndh, self.mean_accuracy_list_dh, self.conventional_accuracy_list_dh
            )

        mean_markers = dict(marker='D', markerfacecolor=colors['black'], markersize=5, markeredgewidth=0)
        meadian_markers = dict(color=colors['black_grey'])

        fig, ax = plt.subplots(figsize=(12, 6))

        box_conventional = ax.boxplot(self.conventioanl_accuracy_list_ndh, positions=[1], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers)
        box_mean = ax.boxplot(self.mean_accuracy_list_ndh, positions=[2], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers)
        box_optimal = ax.boxplot(self.optimal_accuracy_list_ndh, positions=[3], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers)
        box_conventional_dh = ax.boxplot(self.conventional_accuracy_list_dh, positions=[4], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers)
        box_mean_dh = ax.boxplot(self.mean_accuracy_list_dh, positions=[5], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers)

        # Set box colors
        for box in box_conventional['boxes']:
            box.set(facecolor=colors['affected'], alpha=0.7)
        for box in box_mean['boxes']:
            box.set(facecolor=colors['affected'], alpha=0.7)
        for box in box_optimal['boxes']:
            box.set(facecolor=colors['affected'], alpha=0.7)
        for box in box_conventional_dh['boxes']:
            box.set(facecolor=colors['healthy'], alpha=0.7)
        for box in box_mean_dh['boxes']:
            box.set(facecolor=colors['healthy'], alpha=0.7)

        # Accuracy is clinically useful (≥90%) according to [Lang et al., 2020]
        ax.axhline(y=0.9, color=colors['pink'], linestyle='dotted', label='Clinically required performance', lw=2.0)
        # random classifier
        ax.axhline(y=0.5, color=colors['black_grey'], linestyle='--', label='Performance of random classifier')
        ax.add_artist(plt.legend(loc='upper right'))

        # Add colors of healthy and affected boxes to legend
        legend_colors = [colors['healthy'], colors['affected']]
        legend_labels = ['Healthy side', 'Affected side']
        legend_patches = [mpatches.Patch(color=color, label=label, alpha=0.7) for color, label in zip(legend_colors, legend_labels)]
        ax.add_artist(plt.legend(handles=legend_patches, loc='lower right'))
        
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xticklabels(['Conventional', 'Mean personalized', 'Personalized', 'Conventional', 'Mean personalized'])
        ax.set_ylim(0.45, 1.0)
        ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.set_yticklabels(['50%', '60%', '70%', '80%', '90%', '100%'])        
        
        plt.ylabel('Accuracy')
        plt.title('Leave one Participant out Cross Validation')

        # Define significance bracket positions, p-values, and heights
        bracket_heights = [0.6, 0.66, 0.63, 0.9]  # Different heights for the brackets above the boxplot
        bracket_positions = [(1, 3), (2, 3), (1, 2), (4, 5)]  # (start, end) of the brackets
        p_values = [ttest_pvalue_optimal_conventional, ttest_pvalue_optimal_mean, ttest_pvalue_conventional_mean, ttest_pvalue_mean_conventional_dh]
        self.plot_significance_brackets(ax, bracket_positions, p_values, bracket_heights, position="below")

        plt.show()

    #TODO F1
    def plot_LOOCV_F1(self):
        colors = thesis_style.get_thesis_colours()
        ttest_pvalue_optimal_conventional, ttest_pvalue_optimal_mean, ttest_pvalue_conventional_mean, ttest_pvalue_mean_conventional_dh = self.check_ANOVA_ttest_Wilcoxon(
            self.optimal_F1_list_ndh, self.conventioanl_F1_list_ndh, self.mean_F1_list_ndh, self.mean_F1_list_dh, self.conventional_F1_list_dh
            )

        mean_markers = dict(marker='D', markerfacecolor=colors['black'], markersize=5, markeredgewidth=0)
        meadian_markers = dict(color=colors['black_grey'])

        fig, ax = plt.subplots(figsize=(12, 6))

        box_conventional = ax.boxplot(self.conventioanl_F1_list_ndh, positions=[1], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers)
        box_mean = ax.boxplot(self.mean_F1_list_ndh, positions=[2], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers)
        box_optimal = ax.boxplot(self.optimal_F1_list_ndh, positions=[3], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers)
        box_conventional_dh = ax.boxplot(self.conventional_F1_list_dh, positions=[4], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers)
        box_mean_dh = ax.boxplot(self.mean_F1_list_dh, positions=[5], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers)

        # Set box colors
        for box in box_conventional['boxes']:
            box.set(facecolor=colors['affected'], alpha=0.7)
        for box in box_mean['boxes']:
            box.set(facecolor=colors['affected'], alpha=0.7)
        for box in box_optimal['boxes']:
            box.set(facecolor=colors['affected'], alpha=0.7)
        for box in box_conventional_dh['boxes']:
            box.set(facecolor=colors['healthy'], alpha=0.7)
        for box in box_mean_dh['boxes']:
            box.set(facecolor=colors['healthy'], alpha=0.7)

        # Add colors of healthy and affected boxes to legend
        legend_colors = [colors['healthy'], colors['affected']]
        legend_labels = ['Healthy side', 'Affected side']
        legend_patches = [mpatches.Patch(color=color, label=label, alpha=0.7) for color, label in zip(legend_colors, legend_labels)]
        ax.add_artist(plt.legend(handles=legend_patches, loc='lower right'))

        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xticklabels(['Conventional', 'Mean personalized', 'Personalized', 'Conventional', 'Mean personalized'])
        ax.set_ylim(-0.1, 1.0)

        plt.ylabel('F1 score')
        plt.title('Leave one Participant out Cross Validation')

        # Define significance bracket positions, p-values, and heights
        bracket_heights = [0.9, 0.85, 0.8, 0.4]  # Different heights for the brackets above the boxplot
        bracket_positions = [(1, 3), (2, 3), (1, 2), (4, 5)]  # (start, end) of the brackets
        p_values = [ttest_pvalue_optimal_conventional, ttest_pvalue_optimal_mean, ttest_pvalue_conventional_mean, ttest_pvalue_mean_conventional_dh]
        self.plot_significance_brackets(ax, bracket_positions, p_values, bracket_heights, position="above")

        plt.show()

    def combine_ndh_dh(self):
        combined_dict = {}
        for key, value in self.optimal_accuracy_perTask.items():
            value_dh = self.mean_accuracy_perTask_dh[key]
            combined_dict[key] = value_dh+value
        return combined_dict       


    def plot_LOOCV_Accuracy_perTask(self):
        colors = thesis_style.get_thesis_colours()
        plt.rcParams.update({'font.size': 16})
        fig, ax = plt.subplots(figsize=(24, 8))

        # Number of tasks
        num_tasks = len(self.optimal_accuracy_perTask)
        
        # Prepare the positions for the boxplots
        position = np.arange(1, num_tasks * 2, 2)

        # Set colors for the boxplots
        mean_markers = dict(marker='D', markerfacecolor=colors['black'], markersize=7, markeredgewidth=0)
        meadian_markers = dict(color=colors['black_grey'])
        box_optimal = ax.boxplot(self.combine_ndh_dh().values(), positions=position, widths=0.8, showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers, showfliers=False)
        for box in box_optimal['boxes']:
            box.set(facecolor=colors['white'], alpha=0.7)
        # Show all data points for each task/box
        label_ploted = False
        for i, task in enumerate(self.combine_ndh_dh().keys()):
            plt.scatter(np.random.normal(position[i], 0.1, size=len(self.optimal_accuracy_perTask[task])), self.optimal_accuracy_perTask[task], zorder=3.1, color=colors['affected'], label='Affected side' if not label_ploted else None)
            plt.scatter(np.random.normal(position[i], 0.1, size=len(self.mean_accuracy_perTask_dh[task])), self.mean_accuracy_perTask_dh[task], zorder=3.0, color=colors['healthy'], label='Healthy side' if not label_ploted else None)
            label_ploted = True
        ax.add_artist(plt.legend(loc='lower left'))

        # Accuracy is clinically useful (≥90%) according to [Lang et al., 2020]
        ax.axhline(y=0.9, color=colors['pink'], linestyle='dotted', linewidth=3, label='Clinically required performance')
        ax.axhline(y=0.5, color=colors['black_grey'], linestyle='--', linewidth=2, label='Performance of random classifier')
        ax.add_artist(plt.legend(handles=[plt.Line2D([], [], color=colors['pink'], linestyle='--', label='Clinically required performance'),
                   plt.Line2D([], [], color=colors['black_grey'], linestyle='--', label='Performance of random classifier')], loc='lower center'))
        
        # Adjust x-tick labels to be in between the grouped boxplots
        ax.set_xticks(position)
        labels = ['pour\nglas', 'drink', 'fold\nrags', 'sort\ndocuments', 'brooming', 'coat', 'keyboard\ntyping', 'stapling', 'walking', 
                          'door', 'resting', 'other', 'wipe\ntable','light\nswitch']
        ax.set_xticklabels(labels)
        plt.xticks(rotation=0, ha='center')
        
        plt.ylabel('Accuracy')
        plt.title('Leave one Participant out Cross Validation per Task')
        
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])  

        plt.tight_layout()
        plt.show()


    def spearman_correlation_classification_impairment(self):
        # Calculate the Spearman correlation
        spearman_correlation_dict = {
            'optimal_YI': spearmanr(self.evaluation_FMA, self.optimal_YI_list_ndh),
            'conventional_YI': spearmanr(self.evaluation_FMA, self.conventioanl_YI_list_ndh),
            'optimal_accuracy': spearmanr(self.evaluation_FMA, self.optimal_accuracy_list_ndh),
            'conventional_accuracy': spearmanr(self.evaluation_FMA, self.conventioanl_accuracy_list_ndh)
        }
        print(spearman_correlation_dict)

    def plot_FMA_scatter(self):
        plt.rcParams.update({'font.size': 16})
        fig, ax = plt.subplots(figsize=(12, 8))
        
        optimal_YI_std = np.std(self.optimal_YI_list_ndh)
        conv_YI_std = np.std(self.conventioanl_YI_list_ndh)
        mean_YI_std = np.std(self.mean_YI_list_ndh)
        ax.scatter(self.evaluation_FMA, self.optimal_YI_list_ndh, label=f'Personalized YI (std: {optimal_YI_std})', color=thesis_style.get_thesis_colours()['dark_blue'], marker='x')
        ax.scatter(self.evaluation_FMA, self.conventioanl_YI_list_ndh, label=f'Conventional YI (std: {conv_YI_std})', color=thesis_style.get_thesis_colours()['light_blue'], marker='x')
        ax.scatter(self.evaluation_FMA, self.mean_YI_list_ndh, label=f'Mean optimal YI (std: {mean_YI_std})', color=thesis_style.get_thesis_colours()['turquoise'], marker='x')
        optimal_AUC_std = np.std(self.optimal_AUC_list_ndh)
        conv_AUC_std = np.std(self.conventioanl_AUC_list_ndh)
        mean_AUC_std = np.std(self.mean_AUC_list_ndh)
        ax.scatter(self.evaluation_FMA, self.optimal_AUC_list_ndh, label=f'Personalized AUV (std: {optimal_AUC_std})', color=thesis_style.get_thesis_colours()['dark_blue'], marker='o')
        ax.scatter(self.evaluation_FMA, self.conventioanl_AUC_list_ndh, label=f'Conventional AUV (std: {conv_AUC_std})', color=thesis_style.get_thesis_colours()['light_blue'], marker='o')
        ax.scatter(self.evaluation_FMA, self.mean_AUC_list_ndh, label=f'Mean optimal AUV (std: {mean_AUC_std})', color=thesis_style.get_thesis_colours()['turquoise'], marker='o')
        optimal_accuracy_std = np.std(self.optimal_accuracy_list_ndh)
        conv_accuracy_std = np.std(self.conventioanl_accuracy_list_ndh)
        mean_accuracy_std = np.std(self.mean_accuracy_list_ndh)
        ax.scatter(self.evaluation_FMA, self.optimal_accuracy_list_ndh, label=f'Personalized Accuracy (std: {optimal_accuracy_std})', color=thesis_style.get_thesis_colours()['dark_blue'], marker='s')
        ax.scatter(self.evaluation_FMA, self.conventioanl_accuracy_list_ndh, label=f'Conventional Accuracy (std: {conv_accuracy_std})', color=thesis_style.get_thesis_colours()['light_blue'], marker='s')
        ax.scatter(self.evaluation_FMA, self.mean_accuracy_list_ndh, label=f'Mean optimal Accuracy (std: {mean_accuracy_std})', color=thesis_style.get_thesis_colours()['turquoise'], marker='s')

        plt.ylabel('Classification Performance')
        plt.xlabel('Fugl-Meyer Assessment Upper Extremity Score')
        plt.title('Classification Performance accross Fugl-Meyer Upper Extremity Score')
        plt.legend()
        plt.show()