import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from scipy.stats import spearmanr, shapiro, f_oneway, ttest_rel, wilcoxon

from utilities import *
from functions.individual_analysis_gmac_function import get_prediction_gmac, AUC_analisys, accuracy_analisys, get_classification_metrics
from functions.statistics import RegressionModel

class LOOCV_performance:
    def __init__(self, json_files):

        participant_data = extract_fields_from_json_files(json_files, ['individual_GMAC_thresholds_affected_side', 'individual_GMAC_thresholds_unaffected_side', 'ARAT_score', 'FMA-UE_score', 'participant_id'])

        self.PARTICIPANT_ID = participant_data['participant_id']
        self.ARAT = participant_data['ARAT_score']
        self.FMA_UE = participant_data['FMA-UE_score']

        individual_thresholds = participant_data['individual_GMAC_thresholds_affected_side']
        self.COUNT_THRESHOLD_NDH = individual_thresholds[:,0]
        self.PITCH_THRESHOLD_NDH = individual_thresholds[:,1]
        individual_thresholds = participant_data['individual_GMAC_thresholds_unaffected_side']
        self.COUNT_THRESHOLD_DH = individual_thresholds[:,0]
        self.PITCH_THRESHOLD_DH = individual_thresholds[:,1]

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

    def get_threshold_model(self, X_train, count_threshold_model='linear', elevation_threshold_model='polynomial', side='NDH'):
        FMA_array = np.array([participant['FMA_UE'] for participant in X_train])
        
        count_threshold_array = np.array([participant[f'COUNT_THRESHOLD_{side}'] for participant in X_train])
        elevation_threshold_array = np.array([participant[f'PITCH_THRESHOLD_{side}'] for participant in X_train])

        regression_model_count = RegressionModel(FMA_array, count_threshold_array)
        regression_model_elevation = RegressionModel(FMA_array, elevation_threshold_array)

        return self.fit_desired_model(regression_model_count, count_threshold_model), self.fit_desired_model(regression_model_elevation, elevation_threshold_model)

    def retreive_personalized_thresholds(self, X_test, count_threshold_model, elevation_threshold_model, side='NDH'):
        assert len(X_test) == 1, "X_test should contain only one participant"
        FMA = X_test[0]['FMA_UE']
        count_predict = count_threshold_model.predict_linear(FMA)#TODO hardcoded to linear
        elevation_predict = elevation_threshold_model.predict_polynomial(FMA, 2)

        print("Validation on participant ", X_test[0]['participant_id'], f" {side}")
        print("Linear Predictions of personal COUNT threshold: ", count_predict, f". Ground truth individual threshold {side}: ", X_test[0][f'COUNT_THRESHOLD_{side}'])
        print("Polynomial Predictions of personal ELEVATION threshold: ", elevation_predict, f". Ground truth individual threshold {side}: ", X_test[0][f'PITCH_THRESHOLD_{side}'])

        return count_predict, elevation_predict
    
    def retreive_mean_thresholds(self, X_train, decision_mode='Subash', side='NDH'):
        count_threshold_array = np.array([participant['COUNT_THRESHOLD_NDH'] for participant in X_train])
        elevation_threshold_array = np.array([participant['PITCH_THRESHOLD_NDH'] for participant in X_train])
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

        self.individual_YI_list_ndh = []
        self.personalized_YI_list_ndh = []
        self.conventional_YI_list_ndh = []
        self.mean_YI_list_ndh = []
        self.individual_YI_list_dh = []
        self.personalized_YI_list_dh = []
        self.conventional_YI_list_dh = []
        self.mean_YI_list_dh = []

        self.individual_accuracy_list_ndh = []
        self.personalized_accuracy_list_ndh = []
        self.conventioanl_accuracy_list_ndh = []
        self.mean_accuracy_list_ndh = []
        self.individual_accuracy_list_dh = []
        self.personalized_accuracy_list_dh = []
        self.conventional_accuracy_list_dh = []
        self.mean_accuracy_list_dh = []

        self.individual_AUC_list_ndh = []
        self.personalized_AUC_list_ndh = []
        self.conventional_AUC_list_ndh = []
        self.mean_AUC_list_ndh = []
        self.individual_AUC_list_dh = []
        self.personalized_AUC_list_dh = []
        self.conventional_AUC_list_dh = []
        self.mean_AUC_list_dh = []

        self.individual_F1_list_ndh = []
        self.personalized_F1_list_ndh = []
        self.conventioanl_F1_list_ndh = []
        self.mean_F1_list_ndh = []
        self.individual_F1_list_dh = []
        self.personalized_F1_list_dh = []
        self.conventional_F1_list_dh = []
        self.mean_F1_list_dh = []

        self.personalized_accuracy_perTask = {}
        self.conventional_accuracy_perTask = {}
        self.mean_accuracy_perTask_dh = {}

        for train_index, test_index in looCV.split(X, y_ndh, groups=group_IDs):
            
            X_train, _ = [X[i] for i in train_index], [y_ndh[i] for i in train_index]
            X_test, y_test = [X[i] for i in test_index], [y_ndh[i] for i in test_index]
            y_test_dh = [y_dh[i] for i in test_index]

            youden_index_individual_ndh, accuracy_individual_ndh, auc_individual_ndh, F1_individual_ndh = self.calculate_GMAC_classification_performance(X_test, y_test, X_test[0]['COUNT_THRESHOLD_NDH'], X_test[0]['PITCH_THRESHOLD_NDH'], side='NDH')

            regression_model_count_ndh, regression_model_elevation_ndh = self.get_threshold_model(X_train)
            personalized_count_threshold_ndh, personalized_elevation_threshold_ndh = self.retreive_personalized_thresholds(X_test, regression_model_count_ndh, regression_model_elevation_ndh)
            youden_index_personalized_ndh, accuracy_personalized_ndh, auc_personalized_ndh, F1_personalized_ndh = self.calculate_GMAC_classification_performance(X_test, y_test, personalized_count_threshold_ndh, personalized_elevation_threshold_ndh, side='NDH')
            
            youden_index_conventional_ndh, accuracy_conventional_ndh, auc_conventional_ndh, F1_conventional_ndh = self.calculate_GMAC_classification_performance(X_test, y_test, 0, 30, side='NDH')

            loocv_mean_count_ndh, loocv_mean_elevation_ndh = self.retreive_mean_thresholds(X_train)
            youden_index_mean_ndh, accuracy_mean_ndh, auc_mean_ndh, F1_mean_ndh = self.calculate_GMAC_classification_performance(X_test, y_test, loocv_mean_count_ndh, loocv_mean_elevation_ndh, side='NDH')

            youden_index_individual_dh, accuracy_individual_dh, auc_individual_dh, F1_individual_dh = self.calculate_GMAC_classification_performance(X_test, y_test_dh, X_test[0]['COUNT_THRESHOLD_DH'], X_test[0]['PITCH_THRESHOLD_DH'], side='DH')

            regression_model_count_dh, regression_model_elevation_dh = self.get_threshold_model(X_train, side='DH')
            personalized_count_threshold_dh, personalized_elevation_threshold_dh = self.retreive_personalized_thresholds(X_test, regression_model_count_dh, regression_model_elevation_dh, side='DH')
            youden_index_personalized_dh, accuracy_personalized_dh, auc_personalized_dh, F1_personalized_dh = self.calculate_GMAC_classification_performance(X_test, y_test_dh, personalized_count_threshold_dh, personalized_elevation_threshold_dh, side='DH')

            youden_index_conventional_dh, accuracy_conventional_dh, auc_conventional_dh, F1_conventional_dh = self.calculate_GMAC_classification_performance(X_test, y_test_dh, 0, 30, side='DH')
            
            loocv_mean_count_dh, loocv_mean_elevation_dh = self.retreive_mean_thresholds(X_train, side='DH')
            youden_index_mean_dh, accuracy_mean_dh, auc_mean_dh, F1_mean_dh = self.calculate_GMAC_classification_performance(X_test, y_test_dh, loocv_mean_count_dh, loocv_mean_elevation_dh, side='DH')

            self.evaluation_FMA.append(X_test[0]['FMA_UE'])

            self.individual_YI_list_ndh.append(youden_index_individual_ndh)
            self.personalized_YI_list_ndh.append(youden_index_personalized_ndh)
            self.conventional_YI_list_ndh.append(youden_index_conventional_ndh)
            self.mean_YI_list_ndh.append(youden_index_mean_ndh)
            self.individual_YI_list_dh.append(youden_index_individual_dh)
            self.personalized_YI_list_dh.append(youden_index_personalized_dh)
            self.conventional_YI_list_dh.append(youden_index_conventional_dh)
            self.mean_YI_list_dh.append(youden_index_mean_dh)

            self.individual_accuracy_list_ndh.append(accuracy_individual_ndh)
            self.personalized_accuracy_list_ndh.append(accuracy_personalized_ndh)
            self.conventioanl_accuracy_list_ndh.append(accuracy_conventional_ndh)
            self.mean_accuracy_list_ndh.append(accuracy_mean_ndh)
            self.individual_accuracy_list_dh.append(accuracy_individual_dh)
            self.personalized_accuracy_list_dh.append(accuracy_personalized_dh)
            self.conventional_accuracy_list_dh.append(accuracy_conventional_dh)
            self.mean_accuracy_list_dh.append(accuracy_mean_dh)

            self.individual_AUC_list_ndh.append(auc_individual_ndh)
            self.personalized_AUC_list_ndh.append(auc_personalized_ndh)
            self.conventional_AUC_list_ndh.append(auc_conventional_ndh)
            self.mean_AUC_list_ndh.append(auc_mean_ndh)
            self.individual_AUC_list_dh.append(auc_individual_dh)
            self.personalized_AUC_list_dh.append(auc_personalized_dh)
            self.conventional_AUC_list_dh.append(auc_conventional_dh)
            self.mean_AUC_list_dh.append(auc_mean_dh)

            self.individual_F1_list_ndh.append(F1_individual_ndh)
            self.personalized_F1_list_ndh.append(F1_personalized_ndh)
            self.conventioanl_F1_list_ndh.append(F1_conventional_ndh)
            self.mean_F1_list_ndh.append(F1_mean_ndh)
            self.individual_F1_list_dh.append(F1_individual_dh)
            self.personalized_F1_list_dh.append(F1_personalized_dh)
            self.conventional_F1_list_dh.append(F1_conventional_dh)
            self.mean_F1_list_dh.append(F1_mean_dh)
            #TODO plot predicted and ground truth personalized count threshold and elevation threshold

            if perTask:
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
            if task in self.personalized_accuracy_perTask:
                self.personalized_accuracy_perTask[task].append(accuracy)
            else:
                self.personalized_accuracy_perTask[task] = [accuracy]
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
    
    def check_ANOVA_ttest_Wilcoxon(self, personalized_distribution, conventional_distribution, mean_distribution, mean_distribution_dh=None, conventional_distribution_dh=None, personalized_distribution_dh=None):
        """
        Check the statistical significance of the differences between the classification performance of the different GMAC thresholds applied using ANOVA, paired t-test, and Wilcoxon signed-rank test.
        Parameters:
        - personalized_distribution (array-like): The distribution of classification performance for the personalized thresholds.
        - conventional_distribution (array-like): The distribution of classification performance for the conventional thresholds.
        - mean_distribution (array-like): The distribution of classification performance for the mean count and elevation thresholds.
        - mean_distribution_dh (array-like, optional): The distribution of classification performance for the mean count and elevation thresholds for the dominant hand. Default is None.
        - conventional_distribution_dh (array-like, optional): The distribution of classification performance for the conventional thresholds for the dominant hand. Default is None.
        - personalized_distribution_dh (array-like, optional): The distribution of classification performance for the personalized thresholds for the dominant hand. Default is None.
        Note:
        - The null hypothesis for the Shapiro-Wilk test is that the data is normally distributed.
        - The null hypothesis for the one-way ANOVA is that there is no significant difference between the groups.
        - The null hypothesis for the paired t-tests is that there is no significant difference between the paired samples.
        - The null hypothesis for the Wilcoxon signed-rank tests is that there is no significant difference between the paired samples.
        """
        # Check for normality using Shapiro-Wilk test
        _, p_personalized = shapiro(personalized_distribution)
        _, p_conventional = shapiro(conventional_distribution)
        _, p_mean = shapiro(mean_distribution)
        _, p_mean_dh = shapiro(mean_distribution_dh)
        _, p_conventional_dh = shapiro(conventional_distribution_dh)
        _, p_personalized_dh = shapiro(personalized_distribution_dh)
        # Perform one-way ANOVA
        _, p_value = f_oneway(personalized_distribution, conventional_distribution, mean_distribution, mean_distribution_dh, conventional_distribution_dh, personalized_distribution_dh)

        if p_personalized > 0.05 and p_conventional > 0.05 and p_mean > 0.05 and p_mean_dh > 0.05 and p_conventional_dh > 0.05 and p_personalized_dh > 0.05:
            print("The data is normally distributed.")
        else:
            print("The data is not normally distributed.")
        if p_value < 0.05:
            print("There is a significant ANOVA difference between the groups.")
        else:
            print("There is no significant ANOVA difference between the groups.")
        
        # Perform paired t-test
        _, p_value_personalized_conventional = ttest_rel(personalized_distribution, conventional_distribution)
        _, p_value_personalized_mean = ttest_rel(personalized_distribution, mean_distribution)
        _, p_value_conventional_mean = ttest_rel(conventional_distribution, mean_distribution)
        _, p_value_mean_conventional_dh = ttest_rel(mean_distribution_dh, conventional_distribution_dh)
        _, p_value_personalized_conventional_dh = ttest_rel(personalized_distribution_dh, conventional_distribution_dh)
        _, p_value_personalized_mean_dh = ttest_rel(personalized_distribution_dh, mean_distribution_dh)
        print(f"Paired t-test personalized vs conventional: {p_value_personalized_conventional}")
        print(f"Paired t-test personalized vs mean: {p_value_personalized_mean}")
        print(f"Paired t-test conventional vs mean: {p_value_conventional_mean}")
        print(f"Paired t-test mean vs conventional DH: {p_value_mean_conventional_dh}")
        print(f"Paired t-test personalized vs conventional DH: {p_value_personalized_conventional_dh}")
        print(f"Paired t-test personalized vs mean DH: {p_value_personalized_mean_dh}")
        
        # Perform Wilcoxon signed-rank test
        _, p_value_personalized_conventional_wilcoxon = wilcoxon(personalized_distribution, conventional_distribution)
        _, p_value_personalized_mean_wilcoxon = wilcoxon(personalized_distribution, mean_distribution)
        _, p_value_conventional_mean_wilcoxon = wilcoxon(conventional_distribution, mean_distribution)
        _, p_value_mean_conventional_dh_wilcoxon = wilcoxon(mean_distribution_dh, conventional_distribution_dh)
        _, p_value_personalized_conventional_dh_wilcoxon = wilcoxon(personalized_distribution_dh, conventional_distribution_dh)
        _, p_value_personalized_mean_dh_wilcoxon = wilcoxon(personalized_distribution_dh, mean_distribution_dh)
        print(f"Wilcoxon signed-rank test personalized vs conventional: {p_value_personalized_conventional_wilcoxon}")
        print(f"Wilcoxon signed-rank test personalized vs mean: {p_value_personalized_mean_wilcoxon}")
        print(f"Wilcoxon signed-rank test conventional vs mean: {p_value_conventional_mean_wilcoxon}")
        print(f"Wilcoxon signed-rank test mean vs conventional DH: {p_value_mean_conventional_dh_wilcoxon}")
        print(f"Wilcoxon signed-rank test personalized vs conventional DH: {p_value_personalized_conventional_dh_wilcoxon}")
        print(f"Wilcoxon signed-rank test personalized vs mean DH: {p_value_personalized_mean_dh_wilcoxon}")

        return p_value_personalized_conventional, p_value_personalized_mean, p_value_conventional_mean, p_value_mean_conventional_dh, p_value_personalized_conventional_dh, p_value_personalized_mean_dh

    def check_bonferroni_wilcoxon(self, individual_distribution, conventional_distribution, mean_distribution, mean_distribution_dh=None, conventional_distribution_dh=None, individual_distribution_dh=None):
        """
        Check the statistical significance of the differences between the classification performance of the different GMAC thresholds applied using Wilcoxon signed-rank test with Bonferroni correction.
        Parameters:
        - individual_distribution (array-like): The distribution of classification performance for the individual-optimized thresholds.
        - conventional_distribution (array-like): The distribution of classification performance for the conventional thresholds.
        - mean_distribution (array-like): The distribution of classification performance for the mean count and elevation thresholds.
        - mean_distribution_dh (array-like, optional): The distribution of classification performance for the mean count and elevation thresholds for the dominant hand. Default is None.
        - conventional_distribution_dh (array-like, optional): The distribution of classification performance for the conventional thresholds for the dominant hand. Default is None.
        - individual_distribution_dh (array-like, optional): The distribution of classification performance for the individual-optimized thresholds for the dominant hand. Default is None.
        Note:
        - The null hypothesis for the Wilcoxon signed-rank tests is that there is no significant difference between the paired samples.
        - Bonferroni correction is applied to account for multiple comparisons.
        """
        _, p_value_individual_conventional = wilcoxon(individual_distribution, conventional_distribution)
        _, p_value_individual_mean = wilcoxon(individual_distribution, mean_distribution)
        _, p_value_conventional_mean = wilcoxon(conventional_distribution, mean_distribution)
        _, p_value_mean_conventional_dh = wilcoxon(mean_distribution_dh, conventional_distribution_dh)
        _, p_value_individual_conventional_dh = wilcoxon(individual_distribution_dh, conventional_distribution_dh)
        _, p_value_individual_mean_dh = wilcoxon(individual_distribution_dh, mean_distribution_dh)
        print(f"Wilcoxon signed-rank test individual-optimized vs conventional: {p_value_individual_conventional}")
        print(f"Wilcoxon signed-rank test individual-optimized vs population-optimized: {p_value_individual_mean}")
        print(f"Wilcoxon signed-rank test conventional vs population-optimized: {p_value_conventional_mean}")
        print(f"Wilcoxon signed-rank test population-optimized vs conventional DH: {p_value_mean_conventional_dh}")
        print(f"Wilcoxon signed-rank test individual-optimized vs conventional DH: {p_value_individual_conventional_dh}")
        print(f"Wilcoxon signed-rank test individual-optimized vs population-optimized DH: {p_value_individual_mean_dh}")
        p_value_individual_conventional_bonferroni = min(p_value_individual_conventional * 3, 1.0)
        p_value_individual_mean_bonferroni = min(p_value_individual_mean * 3, 1.0)
        p_value_conventional_mean_bonferroni = min(p_value_conventional_mean * 3, 1.0)
        p_value_mean_conventional_dh_bonferroni = min(p_value_mean_conventional_dh * 3, 1.0)
        p_value_individual_conventional_dh_bonferroni = min(p_value_individual_conventional_dh * 3, 1.0)
        p_value_individual_mean_dh_bonferroni = min(p_value_individual_mean_dh * 3, 1.0)
        print(f"Bonferroni corrected p-value individual-optimized vs conventional: {p_value_individual_conventional_bonferroni}")
        print(f"Bonferroni corrected p-value individual-optimized vs population-optimized: {p_value_individual_mean_bonferroni}")
        print(f"Bonferroni corrected p-value conventional vs population-optimized: {p_value_conventional_mean_bonferroni}")
        print(f"Bonferroni corrected p-value population-optimized vs conventional DH: {p_value_mean_conventional_dh_bonferroni}")
        print(f"Bonferroni corrected p-value individual-optimized vs conventional DH: {p_value_individual_conventional_dh_bonferroni}")
        print(f"Bonferroni corrected p-value individual-optimized vs population-optimized DH: {p_value_individual_mean_dh_bonferroni}")
        return p_value_individual_conventional_bonferroni, p_value_individual_mean_bonferroni, p_value_conventional_mean_bonferroni, p_value_mean_conventional_dh_bonferroni, p_value_individual_conventional_dh_bonferroni, p_value_individual_mean_dh_bonferroni

    def print_classification_performance(self, individual_optimized, population_optimized, conventional, population_optimized_dh, conventional_dh, individual_optimized_dh, metric='ROCAUC'):
        """
        Print the classification performance metrics for the different GMAC thresholds.
        Parameters:
        """
        print(f"Individual-optimized GMAC threshold affected {metric} mean: {np.mean(individual_optimized):.2f}, standard deviation: {np.std(individual_optimized):.2f}")
        print(f"Population-optimized GMAC threshold affected {metric} mean: {np.mean(population_optimized):.2f}, standard deviation: {np.std(population_optimized):.2f}")
        print(f"Conventional GMAC threshold affected {metric} mean: {np.mean(conventional):.2f}, standard deviation: {np.std(conventional):.2f}")
        print(f"Individual-optimized GMAC threshold unaffected {metric} mean: {np.mean(individual_optimized_dh):.2f}, standard deviation: {np.std(individual_optimized_dh):.2f}")
        print(f"Population-optimized GMAC threshold unaffected {metric} mean: {np.mean(population_optimized_dh):.2f}, standard deviation: {np.std(population_optimized_dh):.2f}")
        print(f"Conventional GMAC threshold unaffected {metric} mean: {np.mean(conventional_dh):.2f}, standard deviation: {np.std(conventional_dh):.2f}")

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
            ax.text((x1 + x2) * .5, y + 1.2*h if position == "above" else y - 2*h, f"p = {p_val:.4f}", 
                    ha='center', va='bottom', color=col, fontsize=8)
            
    def plot_significance_stars(self, ax, bracket_positions, p_values, bracket_heights, position="above"):
        """
        Adds significance stars to the plot, either above or below the boxplots.
        *(p < 0.05), **(p < 0.01), ***(p < 0.001)
        """
        for i, ((start, end), p_val, y) in enumerate(zip(bracket_positions, p_values, bracket_heights)):
            x1, x2 = start, end  # x-coordinates of the brackets
            h, col = 0.02, 'k'  # Adjust height and color of the bracket
            if x1 is None or x2 is None:
                continue

            # decide whether this bracket's annotation is above or below
            half = len(bracket_positions) // 2
            if position == "below":
                is_above = False
            elif position == "below_above":
                # first half below, second half above
                is_above = (i >= half)
            else:
                is_above = True

            y_text = y + 1.0 * h if is_above else y - 1.5 * h
            va = 'bottom' if is_above else 'top'

            # choose significance stars
            if p_val < 0.001:
                stars = '***'
            elif p_val < 0.01:
                stars = '**'
            elif p_val < 0.05:
                stars = '*'
            else:
                continue
            ax.text((x1 + x2) * .5, y_text, stars, ha='center', va=va, color=col, fontsize=12)

            # Draw brackets
            if position == "above" or (position == "below_above" and i >= len(bracket_positions) // 2):
                ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.0, c=col)
            elif position == "below" or (position == "below_above" and i < len(bracket_positions) // 2):
                ax.plot([x1, x1, x2, x2], [y, y - h, y - h, y], lw=1.0, c=col)

    #TODO combine the following three functions into one
    def plot_LOOCV_YoudenIndex(self):
        colors = thesis_style.get_thesis_colours()
        # Get p-values from the test
        ttest_pvalue_personalized_conventional, ttest_pvalue_personalized_mean, ttest_pvalue_conventional_mean, ttest_pvalue_mean_conventional_dh, ttest_pvalue_personalized_conventional_dh, ttest_pvalue_personalized_mean_dh = self.check_ANOVA_ttest_Wilcoxon(
            self.personalized_YI_list_ndh, self.conventional_YI_list_ndh, self.mean_YI_list_ndh, self.mean_YI_list_dh, self.conventional_YI_list_dh, self.personalized_YI_list_dh
        )
        
        mean_markers = dict(marker='D', markerfacecolor=colors['black'], markersize=5, markeredgewidth=0)
        median_markers = dict(color=colors['black_grey'])

        self.print_classification_performance(self.individual_YI_list_ndh, self.mean_YI_list_ndh, self.conventional_YI_list_ndh, self.mean_YI_list_dh, self.conventional_YI_list_dh, self.individual_YI_list_dh, metric='Youden Index')

        fig, ax = plt.subplots(figsize=(12, 6))

        # Boxplots
        box_conventional = ax.boxplot(self.conventional_YI_list_ndh, positions=[1], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=median_markers, widths=0.3)
        box_mean = ax.boxplot(self.mean_YI_list_ndh, positions=[2], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=median_markers, widths=0.3)
        box_personalized = ax.boxplot(self.personalized_YI_list_ndh, positions=[3], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=median_markers, widths=0.3)
        box_conventional_dh = ax.boxplot(self.conventional_YI_list_dh, positions=[4], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=median_markers, widths=0.3)
        box_mean_dh = ax.boxplot(self.mean_YI_list_dh, positions=[5], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=median_markers, widths=0.3)
        box_personalized_dh = ax.boxplot(self.personalized_YI_list_dh, positions=[6], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=median_markers, widths=0.3)

        # Set box colors
        for box in box_conventional['boxes']:
            box.set(facecolor=colors['affected'], alpha=0.7)
        for box in box_mean['boxes']:
            box.set(facecolor=colors['affected'], alpha=0.7)
        for box in box_personalized['boxes']:
            box.set(facecolor=colors['affected'], alpha=0.7)
        for box in box_conventional_dh['boxes']:
            box.set(facecolor=colors['healthy'], alpha=0.7)
        for box in box_mean_dh['boxes']:
            box.set(facecolor=colors['healthy'], alpha=0.7)
        for box in box_personalized_dh['boxes']:
            box.set(facecolor=colors['healthy'], alpha=0.7)

        # random classifier
        ax.axhline(y=0.0, color=colors['black_grey'], linestyle='--', label='Performance of random classifier', lw=2.0)
        ax.add_artist(plt.legend(loc='lower right'))

        # Add colors of healthy and affected boxes to legend
        legend_colors = [colors['healthy'], colors['affected']]
        legend_labels = ['Unaffected side', 'Affected side']
        legend_patches = [mpatches.Patch(color=color, label=label, alpha=0.7) for color, label in zip(legend_colors, legend_labels)]
        ax.add_artist(plt.legend(handles=legend_patches, loc='upper right', reverse=True))

        # Set x-ticks and labels
        ax.set_xticks([1, 2, 3, 4, 5, 6])
        ax.set_xticklabels(['conventional\nthresholds', 'optimal\nthresholds', 'personalized\nthresholds', 'conventional\nthresholds', 'optimal\nthresholds', 'personalized\nthresholds'], fontsize=10)
        ax.set_ylim(-0.1, 1)

        # Set y-label and title
        plt.ylabel('Youden Index')
        plt.title('GMAC leave one subject out cross validation')

        # Define significance bracket positions, p-values, and heights
        bracket_heights = [0.65, 0.7, 0.75, 0.7, 0.65, 0.75]  # Different heights for the brackets above the boxplot
        bracket_positions = [(1, 3), (2, 3), (1, 2), (4, 5), (4, 6), (5, 6)]  # (start, end) of the brackets
        p_values = [ttest_pvalue_personalized_conventional, ttest_pvalue_personalized_mean, ttest_pvalue_conventional_mean, ttest_pvalue_mean_conventional_dh, 
                    ttest_pvalue_personalized_conventional_dh, ttest_pvalue_personalized_mean_dh]
        self.plot_significance_brackets(ax, bracket_positions, p_values, bracket_heights, position="above")

        plt.show()

    def plot_LOOCV_AUC(self, significance_brackets='pvalues'):
        colors = thesis_style.get_thesis_colours()
        adjwilcoxon_pvalue_individual_conventional, adjwilcoxon_pvalue_individual_mean, adjwilcoxon_pvalue_conventional_mean, adjwilcoxon_pvalue_mean_conventional_dh, adjwilcoxon_pvalue_individual_conventional_dh, adjwilcoxon_pvalue_individual_mean_dh = self.check_bonferroni_wilcoxon(
            self.individual_AUC_list_ndh, self.conventional_AUC_list_ndh, self.mean_AUC_list_ndh, self.mean_AUC_list_dh, self.conventional_AUC_list_dh, self.individual_AUC_list_dh
        )

        mean_markers = dict(marker='D', markerfacecolor=colors['black'], markersize=5.5, markeredgewidth=0)
        meadian_markers = dict(color=colors['black_grey'])

        self.print_classification_performance(self.individual_AUC_list_ndh, self.mean_AUC_list_ndh, self.conventional_AUC_list_ndh,
                                              self.mean_AUC_list_dh, self.conventional_AUC_list_dh, self.individual_AUC_list_dh, metric='ROCAUC')

        fig, ax = plt.subplots(figsize=(12, 5))

        # AUC is clinically useful (≥0.75) according to [Fan et al., 2006]
        ax.axhline(y=0.75, color=colors['grey'], linestyle='dotted', label='Clinically required performance [Fan et al., 2006]', lw=2.0)
        # random classifier
        ax.axhline(y=0.5, color=colors['black_grey'], linestyle='--', label='Random classifier', lw=1.3)
        ax.add_artist(plt.legend(loc='upper left', frameon=False, fontsize=10))

        box_conventional = ax.boxplot(self.conventional_AUC_list_ndh, positions=[1], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers, widths=0.4)
        box_mean = ax.boxplot(self.mean_AUC_list_ndh, positions=[2], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers, widths=0.4)
        box_individual = ax.boxplot(self.individual_AUC_list_ndh, positions=[3], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers, widths=0.4)
        box_conventional_dh = ax.boxplot(self.conventional_AUC_list_dh, positions=[4], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers, widths=0.4)
        box_mean_dh = ax.boxplot(self.mean_AUC_list_dh, positions=[5], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers, widths=0.4)
        box_individual_dh = ax.boxplot(self.individual_AUC_list_dh, positions=[6], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers, widths=0.4)

        # Set box colors
        for box in box_conventional['boxes']:
            box.set(facecolor=colors['affected'], alpha=0.6)
        for box in box_mean['boxes']:
            box.set(facecolor=colors['affected'], alpha=0.6)
        for box in box_individual['boxes']:
            box.set(facecolor=colors['affected'], alpha=0.6)
        for box in box_conventional_dh['boxes']:
            box.set(facecolor=colors['healthy'], alpha=0.6)
        for box in box_mean_dh['boxes']:
            box.set(facecolor=colors['healthy'], alpha=0.6)
        for box in box_individual_dh['boxes']:
            box.set(facecolor=colors['healthy'], alpha=0.6)

        # Add colors of healthy and affected boxes to legend
        legend_colors = [colors['healthy'], colors['affected']]
        legend_labels = ['Unaffected side', 'Affected side']
        legend_patches = [mpatches.Patch(color=color, label=label, alpha=0.6) for color, label in zip(legend_colors, legend_labels)]
        ax.add_artist(plt.legend(handles=legend_patches, frameon=False, loc='upper right', reverse=True))

        ax.set_xticks([1, 2, 3, 4, 5, 6])
        ax.set_xticklabels(['Conventional\nthresholds', 'Population-optimized\nthresholds', 'Individual-optimized\nthresholds', 'Conventional\nthresholds', 'Population-optimized\nthresholds', 'Individual-optimized\nthresholds'], fontsize=10)
        ax.set_ylim(0.45, 1.0)

        plt.rcParams.update({'font.size': 12})
        plt.ylabel('ROC AUC', fontsize=11)
        plt.title('Functional movement detection performance')

        # Define significance bracket positions, p-values, and heights
        bracket_heights = [0.81, 0.9, 0.86, 0.81, 0.9, 0.86]  # Different heights for the brackets above the boxplot
        bracket_positions = [(1, 3), (2, 3), (1, 2), (4, 5), (4, 6), (5, 6)]  # (start, end) of the brackets
        p_values = [adjwilcoxon_pvalue_individual_conventional, adjwilcoxon_pvalue_individual_mean, adjwilcoxon_pvalue_conventional_mean, 
                    adjwilcoxon_pvalue_mean_conventional_dh, adjwilcoxon_pvalue_individual_conventional_dh, adjwilcoxon_pvalue_individual_mean_dh]
        if significance_brackets == 'stars':
            bracket_heights = [0.57, 0.615, 0.83, 0.81, 0.88, 0.84]
            bracket_positions = [(1, 3), (2, 3), (None, None), (4, 5), (4, 6), (5, 6)]
            self.plot_significance_stars(ax, bracket_positions, p_values, bracket_heights, position="below_above")
        else:
            self.plot_significance_brackets(ax, bracket_positions, p_values, bracket_heights, position="above")

        plt.savefig(os.path.join(save_path.downloadsPath, 'ROC_AUC_withPopulationOptimized.pdf'), bbox_inches='tight')
        plt.show()

    def plot_LOOCV_Accuracy(self):
        colors = thesis_style.get_thesis_colours()
        ttest_pvalue_personalized_conventional, ttest_pvalue_personalized_mean, ttest_pvalue_conventional_mean, ttest_pvalue_mean_conventional_dh, ttest_pvalue_personalized_conventional_dh, ttest_pvalue_personalized_mean_dh = self.check_ANOVA_ttest_Wilcoxon(
            self.personalized_accuracy_list_ndh, self.conventioanl_accuracy_list_ndh, self.mean_accuracy_list_ndh, self.mean_accuracy_list_dh, self.conventional_accuracy_list_dh, self.personalized_accuracy_list_dh
            )

        mean_markers = dict(marker='D', markerfacecolor=colors['black'], markersize=5, markeredgewidth=0)
        meadian_markers = dict(color=colors['black_grey'])

        fig, ax = plt.subplots(figsize=(12, 6))

        box_conventional = ax.boxplot(self.conventioanl_accuracy_list_ndh, positions=[1], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers, widths=0.3)
        box_mean = ax.boxplot(self.mean_accuracy_list_ndh, positions=[2], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers, widths=0.3)
        box_personalized = ax.boxplot(self.personalized_accuracy_list_ndh, positions=[3], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers, widths=0.3)
        box_conventional_dh = ax.boxplot(self.conventional_accuracy_list_dh, positions=[4], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers, widths=0.3)
        box_mean_dh = ax.boxplot(self.mean_accuracy_list_dh, positions=[5], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers, widths=0.3)
        box_personalized_dh = ax.boxplot(self.personalized_accuracy_list_dh, positions=[6], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers, widths=0.3)

        # Set box colors
        for box in box_conventional['boxes']:
            box.set(facecolor=colors['affected'], alpha=0.7)
        for box in box_mean['boxes']:
            box.set(facecolor=colors['affected'], alpha=0.7)
        for box in box_personalized['boxes']:
            box.set(facecolor=colors['affected'], alpha=0.7)
        for box in box_conventional_dh['boxes']:
            box.set(facecolor=colors['healthy'], alpha=0.7)
        for box in box_mean_dh['boxes']:
            box.set(facecolor=colors['healthy'], alpha=0.7)
        for box in box_personalized_dh['boxes']:
            box.set(facecolor=colors['healthy'], alpha=0.7)

        # Accuracy is clinically useful (≥90%) according to [Lang et al., 2020]
        ax.axhline(y=0.9, color=colors['pink'], linestyle='dotted', label='Clinically required performance [Lang et al., 2020]', lw=3.0)
        # random classifier
        ax.axhline(y=0.5, color=colors['black_grey'], linestyle='--', label='Performance of random classifier', lw=2.0)
        ax.add_artist(plt.legend(loc='upper right'))

        # Add colors of healthy and affected boxes to legend
        legend_colors = [colors['healthy'], colors['affected']]
        legend_labels = ['Unaffected side', 'Affected side']
        legend_patches = [mpatches.Patch(color=color, label=label, alpha=0.7) for color, label in zip(legend_colors, legend_labels)]
        ax.add_artist(plt.legend(handles=legend_patches, loc='lower right', reverse=True))
        
        ax.set_xticks([1, 2, 3, 4, 5, 6])
        ax.set_xticklabels(['conventional\nthresholds', 'optimal\nthresholds', 'personalized\nthresholds', 'conventional\nthresholds', 'optimal\nthresholds', 'personalized\nthresholds'], fontsize=10)
        ax.set_ylim(0.45, 1.0)
        ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.set_yticklabels(['50%', '60%', '70%', '80%', '90%', '100%'])        
        
        plt.ylabel('Accuracy')
        plt.title('GMAC leave one subject out cross validation')

        # Define significance bracket positions, p-values, and heights
        bracket_heights = [0.6, 0.66, 0.63, 0.9, 0.6, 0.66]  # Different heights for the brackets above the boxplot
        bracket_positions = [(1, 3), (2, 3), (1, 2), (4, 5), (4, 6), (5, 6)]  # (start, end) of the brackets
        p_values = [ttest_pvalue_personalized_conventional, ttest_pvalue_personalized_mean, ttest_pvalue_conventional_mean, 
                    ttest_pvalue_mean_conventional_dh, ttest_pvalue_personalized_conventional_dh, ttest_pvalue_personalized_mean_dh]
        self.plot_significance_brackets(ax, bracket_positions, p_values, bracket_heights, position="below")

        plt.show()

    def plot_LOOCV_F1(self):
        colors = thesis_style.get_thesis_colours()
        ttest_pvalue_personalized_conventional, ttest_pvalue_personalized_mean, ttest_pvalue_conventional_mean, ttest_pvalue_mean_conventional_dh, ttest_pvalue_personalized_conventional_dh, ttest_pvalue_personalized_mean_dh = self.check_ANOVA_ttest_Wilcoxon(
            self.personalized_F1_list_ndh, self.conventioanl_F1_list_ndh, self.mean_F1_list_ndh, self.mean_F1_list_dh, self.conventional_F1_list_dh, self.personalized_F1_list_dh
            )

        mean_markers = dict(marker='D', markerfacecolor=colors['black'], markersize=5, markeredgewidth=0)
        meadian_markers = dict(color=colors['black_grey'])

        fig, ax = plt.subplots(figsize=(12, 6))

        box_conventional = ax.boxplot(self.conventioanl_F1_list_ndh, positions=[1], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers, widths=0.3)
        box_mean = ax.boxplot(self.mean_F1_list_ndh, positions=[2], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers, widths=0.3)
        box_personalized = ax.boxplot(self.personalized_F1_list_ndh, positions=[3], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers, widths=0.3)
        box_conventional_dh = ax.boxplot(self.conventional_F1_list_dh, positions=[4], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers, widths=0.3)
        box_mean_dh = ax.boxplot(self.mean_F1_list_dh, positions=[5], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers, widths=0.3)
        box_personalized_dh = ax.boxplot(self.personalized_F1_list_dh, positions=[6], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers, widths=0.3)

        # Set box colors
        for box in box_conventional['boxes']:
            box.set(facecolor=colors['affected'], alpha=0.7)
        for box in box_mean['boxes']:
            box.set(facecolor=colors['affected'], alpha=0.7)
        for box in box_personalized['boxes']:
            box.set(facecolor=colors['affected'], alpha=0.7)
        for box in box_conventional_dh['boxes']:
            box.set(facecolor=colors['healthy'], alpha=0.7)
        for box in box_mean_dh['boxes']:
            box.set(facecolor=colors['healthy'], alpha=0.7)
        for box in box_personalized_dh['boxes']:
            box.set(facecolor=colors['healthy'], alpha=0.7)

        # Add colors of healthy and affected boxes to legend
        legend_colors = [colors['healthy'], colors['affected']]
        legend_labels = ['Unaffected side', 'Affected side']
        legend_patches = [mpatches.Patch(color=color, label=label, alpha=0.7) for color, label in zip(legend_colors, legend_labels)]
        ax.add_artist(plt.legend(handles=legend_patches, loc='lower right', reverse=True))

        ax.set_xticks([1, 2, 3, 4, 5, 6])
        ax.set_xticklabels(['conventional\nthresholds', 'optimal\nthresholds', 'personalized\nthresholds', 'conventional\nthresholds', 'optimal\nthresholds', 'personalized\nthresholds'], fontsize=10)
        ax.set_ylim(-0.1, 1.0)

        plt.ylabel('F1 score')
        plt.title('GMAC leave one subject out cross validation')

        # Define significance bracket positions, p-values, and heights
        bracket_heights = [0.9, 0.85, 0.8, 0.4, 0.9, 0.85]  # Different heights for the brackets above the boxplot
        bracket_positions = [(1, 3), (2, 3), (1, 2), (4, 5), (4, 6), (5, 6)]  # (start, end) of the brackets
        p_values = [ttest_pvalue_personalized_conventional, ttest_pvalue_personalized_mean, ttest_pvalue_conventional_mean, 
                    ttest_pvalue_mean_conventional_dh, ttest_pvalue_personalized_conventional_dh, ttest_pvalue_personalized_mean_dh]
        self.plot_significance_brackets(ax, bracket_positions, p_values, bracket_heights, position="above")

        plt.show()

    def combine_ndh_dh(self):
        combined_dict = {}
        for key, value in self.personalized_accuracy_perTask.items():
            print(f"Personalized threshold GMAC accuracy affected {key}: {np.mean(value)*100:.2f}%")
            print(f"Fixed for all optimal threshold GMAC accuracy unaffected {key}: {np.mean(self.mean_accuracy_perTask_dh[key])*100:.2f}%")
            value_dh = self.mean_accuracy_perTask_dh[key]
            combined_dict[key] = value_dh+value
        return combined_dict

    def plot_simplified_accuracy_perTask(self, sorted_tasks, combined_sides, sorted_labels, colors):       
        '''
        Plot violin plots showing the the distribution of accuracy per task.
        '''
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(20, 6))

        # Prepare the positions for the violin plots
        position = np.arange(0, len(sorted_tasks), 1)
        # Plot violin plots for each task
        violin_parts = ax.violinplot([combined_sides[task] for task in sorted_tasks], positions=position, widths=0.8, showmeans=False, showmedians=False, showextrema=False)

        # Customize violin plot appearance
        for i, pc in enumerate(violin_parts['bodies']):
            pc.set_facecolor(colors['light_blue'])

        # Plot the means as diamonds on top of the violin plots
        means = [np.mean(combined_sides[task]) for task in sorted_tasks]
        ax.plot(position, means, 'D', color=colors['dark_blue'], markersize=12, label=f'Mean over both sides of {int(len(combined_sides[sorted_tasks[0]])/2)} subjects')

        # Accuracy is clinically useful (≥90%) according to [Lang et al., 2020]
        ax.axhline(y=0.9, color=colors['pink'], linestyle='dotted', linewidth=4, label=f"Accuracy required for clinical implementation [Lang et al., 2020]")

        ax.set_xticks(position)
        ax.set_xticklabels(sorted_labels)
        ax.set_xlim(-0.5, len(sorted_tasks) - 0.5)
        ax.set_ylim(0, 1.02)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
        ax.set_ylabel('Accuracy')

        # Set the plot title
        ax.set_title('GMAC leave one subject out cross validation')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_LOOCV_Accuracy_perTask(self):
        colors = thesis_style.get_thesis_colours()
        plt.rcParams.update({'font.size': 16})
        fig, ax = plt.subplots(figsize=(24, 6))

        num_tasks = len(self.personalized_accuracy_perTask)
        combined_sides = self.combine_ndh_dh()
        
        # Sort the tasks based on mean accuracy
        sorted_tasks = sorted(combined_sides.keys(), key=lambda x: np.mean(combined_sides[x]), reverse=True)
        sorted_labels = [task_to_formated.get_formated_task_labels()[list(combined_sides.keys()).index(task)] for task in sorted_tasks]
        
        # Prepare the positions for the boxplots
        position = np.arange(1, num_tasks * 2, 2)

        # Set colors for the boxplots
        mean_markers = dict(marker='D', markerfacecolor=colors['black'], markersize=10, markeredgewidth=0, zorder=4)
        meadian_markers = dict(color=colors['black_grey'], alpha=0.0)
        box_optimal = ax.boxplot([combined_sides[task] for task in sorted_tasks], positions=position, widths=0.8, showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers, showfliers=False)
        for box in box_optimal['boxes']:
            box.set(facecolor=colors['white'], alpha=0.7)
        # Show all data points for each task/box
        label_ploted = False
        for i, task in enumerate(sorted_tasks):
            plt.scatter(np.random.normal(position[i], 0.1, size=len(self.personalized_accuracy_perTask[task])), self.personalized_accuracy_perTask[task], zorder=3.1, color=colors['affected'], label='Affected side' if not label_ploted else None)
            plt.scatter(np.random.normal(position[i], 0.1, size=len(self.mean_accuracy_perTask_dh[task])), self.mean_accuracy_perTask_dh[task], zorder=3.0, color=colors['healthy'], label='Unaffected side' if not label_ploted else None)
            label_ploted = True
        ax.add_artist(plt.legend(loc='lower left'))

        # Accuracy is clinically useful (≥90%) according to [Lang et al., 2020]
        ax.axhline(y=0.9, color=colors['pink'], linestyle='dotted', linewidth=4, label='Clinically required performance [Lang et al., 2020]')
        #ax.axhline(y=0.5, color=colors['black_grey'], linestyle='--', linewidth=3, label='Performance of random classifier')
        ax.add_artist(plt.legend(handles=[plt.Line2D([], [], marker='D', markersize=8, color=colors['black'], linestyle='', label='Mean accuracy per task'),
            plt.Line2D([], [], color=colors['pink'], linestyle='dotted', lw=4, label='Clinically required performance [Lang et al., 2020]')#, plt.Line2D([], [], color=colors['black_grey'], linestyle='--', lw=3, label='Performance of random classifier')
                   ], loc='lower right'))
        
        # Adjust x-tick labels to be in between the grouped boxplots
        ax.set_xticks(position)
        ax.set_xticklabels(sorted_labels)
        plt.xticks(rotation=0, ha='center')
        
        plt.ylabel('Accuracy')
        plt.title('GMAC leave one subject out cross validation per Task')
        
        ax.set_ylim(0.0, 1.02)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])  

        plt.tight_layout()
        plt.show()

        self.plot_simplified_accuracy_perTask(sorted_tasks, combined_sides, sorted_labels, colors)


    def spearman_correlation_classification_impairment(self):
        # Calculate the Spearman correlation
        spearman_correlation_dict = {
            'personalized_YI': spearmanr(self.evaluation_FMA, self.personalized_YI_list_ndh),
            'conventional_YI': spearmanr(self.evaluation_FMA, self.conventional_YI_list_ndh),
            'personalized_accuracy': spearmanr(self.evaluation_FMA, self.personalized_accuracy_list_ndh),
            'conventional_accuracy': spearmanr(self.evaluation_FMA, self.conventioanl_accuracy_list_ndh)
        }
        print(spearman_correlation_dict)

    def plot_FMA_scatter(self):
        plt.rcParams.update({'font.size': 16})
        fig, ax = plt.subplots(figsize=(12, 8))
        
        optimal_YI_std = np.std(self.personalized_YI_list_ndh)
        conv_YI_std = np.std(self.conventional_YI_list_ndh)
        mean_YI_std = np.std(self.mean_YI_list_ndh)
        ax.scatter(self.evaluation_FMA, self.personalized_YI_list_ndh, label=f'Personalized YI (std: {optimal_YI_std})', color=thesis_style.get_thesis_colours()['dark_blue'], marker='x')
        ax.scatter(self.evaluation_FMA, self.conventional_YI_list_ndh, label=f'Conventional YI (std: {conv_YI_std})', color=thesis_style.get_thesis_colours()['light_blue'], marker='x')
        ax.scatter(self.evaluation_FMA, self.mean_YI_list_ndh, label=f'Mean optimal YI (std: {mean_YI_std})', color=thesis_style.get_thesis_colours()['turquoise'], marker='x')
        optimal_AUC_std = np.std(self.personalized_AUC_list_ndh)
        conv_AUC_std = np.std(self.conventional_AUC_list_ndh)
        mean_AUC_std = np.std(self.mean_AUC_list_ndh)
        ax.scatter(self.evaluation_FMA, self.personalized_AUC_list_ndh, label=f'Personalized AUV (std: {optimal_AUC_std})', color=thesis_style.get_thesis_colours()['dark_blue'], marker='o')
        ax.scatter(self.evaluation_FMA, self.conventional_AUC_list_ndh, label=f'Conventional AUV (std: {conv_AUC_std})', color=thesis_style.get_thesis_colours()['light_blue'], marker='o')
        ax.scatter(self.evaluation_FMA, self.mean_AUC_list_ndh, label=f'Mean optimal AUV (std: {mean_AUC_std})', color=thesis_style.get_thesis_colours()['turquoise'], marker='o')
        optimal_accuracy_std = np.std(self.personalized_accuracy_list_ndh)
        conv_accuracy_std = np.std(self.conventioanl_accuracy_list_ndh)
        mean_accuracy_std = np.std(self.mean_accuracy_list_ndh)
        ax.scatter(self.evaluation_FMA, self.personalized_accuracy_list_ndh, label=f'Personalized Accuracy (std: {optimal_accuracy_std})', color=thesis_style.get_thesis_colours()['dark_blue'], marker='s')
        ax.scatter(self.evaluation_FMA, self.conventioanl_accuracy_list_ndh, label=f'Conventional Accuracy (std: {conv_accuracy_std})', color=thesis_style.get_thesis_colours()['light_blue'], marker='s')
        ax.scatter(self.evaluation_FMA, self.mean_accuracy_list_ndh, label=f'Mean optimal Accuracy (std: {mean_accuracy_std})', color=thesis_style.get_thesis_colours()['turquoise'], marker='s')

        plt.ylabel('Classification Performance')
        plt.xlabel('Fugl-Meyer Assessment Upper Extremity Score')
        plt.title('Classification Performance accross Fugl-Meyer Upper Extremity Score')
        plt.legend()
        plt.show()