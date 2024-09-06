import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
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
        for path in json_files:
            dict_1Hz = extract_fields_from_json_files([path], ['GT_mask_NDH_1Hz', 'GT_mask_DH_1Hz', 'counts_for_GMAC_ndh_1Hz', 'counts_for_GMAC_dh_1Hz', 'task_mask_for_GMAC_NDH_1Hz', 'pitch_for_GMAC_ndh_1Hz', 'pitch_for_GMAC_dh_1Hz'])
            task_mask_ndh_1Hz.append(dict_1Hz['task_mask_for_GMAC_NDH_1Hz'])
            GT_NDH_1Hz.append(dict_1Hz['GT_mask_NDH_1Hz'])
            GT_DH_1Hz.append(dict_1Hz['GT_mask_DH_1Hz'])
            counts_NDH_1Hz.append(dict_1Hz['counts_for_GMAC_ndh_1Hz'])
            counts_DH_1Hz.append(dict_1Hz['counts_for_GMAC_dh_1Hz'])
            elevation_NDH_1Hz.append(dict_1Hz['pitch_for_GMAC_ndh_1Hz'])
            elevation_DH_1Hz.append(dict_1Hz['pitch_for_GMAC_dh_1Hz'])
        task_mask = {'task_mask_ndh_1Hz': task_mask_ndh_1Hz}
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
    
    def retreive_mean_thresholds(self, X_train, decision_mode='Subash'):
        count_threshold_array = np.array([participant['COUNT_THRESHOLD_NDH'] for participant in X_train])
        elevation_threshold_array = np.array([participant['PITCH_THRESHOLD_NDH'] for participant in X_train])
        if decision_mode=='Linus':
            count_threshold_array = np.array([participant['COUNT_THRESHOLD_NDH_Linus'] for participant in X_train])
            elevation_threshold_array = np.array([participant['PITCH_THRESHOLD_NDH_Linus'] for participant in X_train])

        return np.mean(count_threshold_array), np.mean(elevation_threshold_array)

    def calculate_GMAC_classification_performance(self, X_test, y_test, personalized_count_threshold, personalized_elevation_threshold):
        assert len(X_test) == 1, "X_test should contain only one participant"
        assert len(y_test) == 1, "y_test should contain only one participant"
        test_gt = y_test[0].flatten()

        counts_array = np.array(X_test[0]['counts_NDH_1Hz'])
        elevation_array = np.array(X_test[0]['elevation_NDH_1Hz'])

        gmac_prediction = get_prediction_gmac(counts_array, elevation_array, count_threshold=personalized_count_threshold, functional_space=personalized_elevation_threshold, decision_mode='Subash')
        gmac_prediction = gmac_prediction.astype(int)

        accuracy, _, _, youden_index = get_classification_metrics(test_gt, gmac_prediction.flatten())
        auc = AUC_analisys(test_gt, gmac_prediction.flatten())
        accuracy_analisys(accuracy)

        return youden_index, accuracy, auc
    
    def calculate_GMAC_classification_performance_perTask(self, task_dict, personalized_count_threshold, personalized_elevation_threshold):
        accuracy_per_task = {}
        for task_of_interest, task_data in task_dict.items():
            count_for_task = np.array(task_data['count'])
            pitch_for_task = np.array(task_data['elevation'])
            gt_for_task = task_data['gt'].flatten()

            gmac_prediction = get_prediction_gmac(count_for_task, pitch_for_task, count_threshold=personalized_count_threshold, functional_space=personalized_elevation_threshold, decision_mode='Subash')
            gmac_prediction = gmac_prediction.astype(int)

            tn, fp, fn, tp = confusion_matrix(gt_for_task, gmac_prediction, labels=[1, 0]).ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)

            accuracy_per_task[task_of_interest] = accuracy

        return accuracy_per_task


    def LOOCV_complete(self, perTask=True):
        """
        Performs Leave-One-Subject-Out Cross Validation (LOOCV) for classification performance evaluation.
        Args:
            perTask (bool, optional): Flag indicating whether to calculate performance metrics per task. 

        Returns:
            None
        """
        participant_dicts = [] # List with one dictionary per participant containing all the necessary participants data
        for i, participant_id in enumerate(self.PARTICIPANT_ID):
            participant_dict = {
                'participant_id': participant_id,
                'ARAT': self.ARAT[i],
                'FMA_UE': self.FMA_UE[i],
                'COUNT_THRESHOLD_NDH': self.COUNT_THRESHOLD_NDH[i],
                'PITCH_THRESHOLD_NDH': self.PITCH_THRESHOLD_NDH[i],
                'COUNT_THRESHOLD_NDH_Linus': self.COUNT_THRESHOLD_NDH_Linus[i],
                'PITCH_THRESHOLD_NDH_Linus': self.PITCH_THRESHOLD_NDH_Linus[i],
                'counts_NDH_1Hz': self.count_data['counts_NDH_1Hz'][i],
                'elevation_NDH_1Hz': self.pitch_data['elevation_NDH_1Hz'][i],
                'task_mask_ndh_1Hz': self.task_mask['task_mask_ndh_1Hz'][i]
            }
            participant_dicts.append(participant_dict)

        looCV = LeaveOneGroupOut()
        X = participant_dicts
        y = self.gt_functional['GT_mask_NDH_1Hz']
        group_IDs = self.PARTICIPANT_ID

        self.evaluation_FMA = []

        self.optimal_YI_list_ndh = []
        #self.optimal_YI_LinReg_list = []#TODO compare all accuracies of linear and polynomial regression combination, also no need to declare all this here
        self.conventioanl_YI_list_ndh = []
        self.mean_YI_list_ndh = []
        self.optimal_YI_list_ndh_Linus = []
        self.mean_YI_list_ndh_Linus = []

        self.optimal_accuracy_list_ndh = []
        #self.optimal_accuracy_LinReg_list = []
        self.conventioanl_accuracy_list_ndh = []
        self.mean_accuracy_list_ndh = []
        self.optimal_accuracy_list_ndh_Linus = []
        self.mean_accuracy_list_ndh_Linus = []

        self.optimal_AUC_list_ndh = []
        #self.optimal_AUC_LinReg_list = []
        self.conventioanl_AUC_list_ndh = []
        self.mean_AUC_list_ndh = []
        self.optimal_AUC_list_ndh_Linus = []
        self.mean_AUC_list_ndh_Linus = []

        self.optimal_accuracy_perTask_ndh = {}
        self.conventional_accuracy_perTask_ndh = {}

        for train_index, test_index in looCV.split(X, y, groups=group_IDs):
            
            X_train, _ = [X[i] for i in train_index], [y[i] for i in train_index]
            X_test, y_test = [X[i] for i in test_index], [y[i] for i in test_index]

            regression_model_count_ndh, regression_model_elevation_ndh = self.get_threshold_model(X_train)
            regression_model_count_ndh_Linus, regression_model_elevation_ndh_Linus = self.get_threshold_model(X_train, decision_mode='Linus')
            #TODO same for dh
            personalized_count_threshold_ndh, personalized_elevation_threshold_ndh = self.retreive_personalized_thresholds(X_test, regression_model_count_ndh, regression_model_elevation_ndh)
            personalized_count_threshold_ndh_Linus, personalized_elevation_threshold_ndh_Linus = self.retreive_personalized_thresholds(X_test, regression_model_count_ndh_Linus, regression_model_elevation_ndh_Linus)
            #TODO same for dh
            youden_index_optimized_ndh, accuracy_optimized_ndh, auc_optimized_ndh = self.calculate_GMAC_classification_performance(X_test, y_test, personalized_count_threshold_ndh, personalized_elevation_threshold_ndh)
            youden_index_conventional, accuracy_conventional, auc_conventional = self.calculate_GMAC_classification_performance(X_test, y_test, 0, 30)
            youden_index_optimized_ndh_Linus, accuracy_optimized_ndh_Linus, auc_optimized_ndh_Linus = self.calculate_GMAC_classification_performance(X_test, y_test, personalized_count_threshold_ndh_Linus, personalized_elevation_threshold_ndh_Linus)
            #TODO same for dh

            loocv_mean_count_ndh, loocv_mean_elevation_ndh = self.retreive_mean_thresholds(X_train)
            youden_index_mean_ndh, accuracy_mean_ndh, auc_mean_ndh = self.calculate_GMAC_classification_performance(X_test, y_test, loocv_mean_count_ndh, loocv_mean_elevation_ndh)
            loocv_mean_count_ndh_Linus, loocv_mean_elevation_ndh_Linus = self.retreive_mean_thresholds(X_train, decision_mode='Linus')
            youden_index_mean_ndh_Linus, accuracy_mean_ndh_Linus, auc_mean_ndh_Linus = self.calculate_GMAC_classification_performance(X_test, y_test, loocv_mean_count_ndh_Linus, loocv_mean_elevation_ndh_Linus)

            self.evaluation_FMA.append(X_test[0]['FMA_UE'])

            self.optimal_YI_list_ndh.append(youden_index_optimized_ndh)
            self.conventioanl_YI_list_ndh.append(youden_index_conventional)
            self.mean_YI_list_ndh.append(youden_index_mean_ndh)
            self.optimal_YI_list_ndh_Linus.append(youden_index_optimized_ndh_Linus)
            self.mean_YI_list_ndh_Linus.append(youden_index_mean_ndh_Linus)

            self.optimal_accuracy_list_ndh.append(accuracy_optimized_ndh)
            self.conventioanl_accuracy_list_ndh.append(accuracy_conventional)
            self.mean_accuracy_list_ndh.append(accuracy_mean_ndh)
            self.optimal_accuracy_list_ndh_Linus.append(accuracy_optimized_ndh_Linus)
            self.mean_accuracy_list_ndh_Linus.append(accuracy_mean_ndh_Linus)

            self.optimal_AUC_list_ndh.append(auc_optimized_ndh)
            self.conventioanl_AUC_list_ndh.append(auc_conventional)
            self.mean_AUC_list_ndh.append(auc_mean_ndh)
            self.optimal_AUC_list_ndh_Linus.append(auc_optimized_ndh_Linus)
            self.mean_AUC_list_ndh_Linus.append(auc_mean_ndh_Linus)
            #TODO plot predicted and ground truth personalized count threshold and elevation threshold

            if perTask:
                self.LOOCV_perTask(X_test, y_test, personalized_count_threshold_ndh, personalized_elevation_threshold_ndh)

    def LOOCV_perTask(self, X_test, y_test, personalized_count_threshold, personalized_elevation_threshold):
        protocol_tasks = ['open_bottle_and_pour_glass', 'drink', 'fold_rags_towels', 'sort_documents', 'brooming', 'putting_on_and_off_coat', 'keyboard_typing', 'stapling', 'walking', 
                          'open_and_close_door', 'resting', 'other', 'wipe_table','light_switch']
        task_dict = {}
        for task_of_interest in protocol_tasks: #attention: hardcoded for ndh
            count_for_task = extract_all_values_with_label(X_test[0]['counts_NDH_1Hz'], X_test[0]['task_mask_ndh_1Hz'], task_of_interest)
            pitch_for_task = extract_all_values_with_label(X_test[0]['elevation_NDH_1Hz'], X_test[0]['task_mask_ndh_1Hz'], task_of_interest)
            gt_for_task = extract_all_values_with_label(y_test[0], X_test[0]['task_mask_ndh_1Hz'], task_of_interest)
            task_dict[task_of_interest] = {'count': count_for_task, 'elevation': pitch_for_task, 'gt': gt_for_task}

        accuracy_per_task_optimized = self.calculate_GMAC_classification_performance_perTask(task_dict, personalized_count_threshold, personalized_elevation_threshold)
        accuracy_per_task_conventional = self.calculate_GMAC_classification_performance_perTask(task_dict, 0, 30)

        for task, accuracy in accuracy_per_task_optimized.items():
            if task in self.optimal_accuracy_perTask_ndh:
                self.optimal_accuracy_perTask_ndh[task].append(accuracy)
            else:
                self.optimal_accuracy_perTask_ndh[task] = [accuracy]
        for task, accuracy in accuracy_per_task_conventional.items():
            if task in self.conventional_accuracy_perTask_ndh:
                self.conventional_accuracy_perTask_ndh[task].append(accuracy)
            else:
                self.conventional_accuracy_perTask_ndh[task] = [accuracy]
    
    def check_ANOVA_ttest_Wilcoxon(self, optimal_distribution, conventional_distribution, mean_distribution):
        """
        Check the statistical significance of the differences between the classification performance of the different GMAC thresholds applied using ANOVA, paired t-test, and Wilcoxon signed-rank test.
        Parameters:
        - optimal_distribution (array-like): The distribution of classification performance for the optimal thresholds.
        - conventional_distribution (array-like): The distribution of classification performance for the conventional thresholds.
        - mean_distribution (array-like): The distribution of classification performance for the mean count and elevation thresholds.
        Returns:
        - None
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
        # Perform one-way ANOVA
        _, p_value = f_oneway(optimal_distribution, conventional_distribution, mean_distribution)

        if p_optimal > 0.05 and p_conventional > 0.05 and p_mean > 0.05:
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
        print(f"Paired t-test optimal vs conventional: {p_value_optimal_conventional}")
        print(f"Paired t-test optimal vs mean: {p_value_optimal_mean}")
        print(f"Paired t-test conventional vs mean: {p_value_conventional_mean}")
        
        # Perform Wilcoxon signed-rank test
        _, p_value_optimal_conventional_wilcoxon = wilcoxon(optimal_distribution, conventional_distribution)
        _, p_value_optimal_mean_wilcoxon = wilcoxon(optimal_distribution, mean_distribution)
        _, p_value_conventional_mean_wilcoxon = wilcoxon(conventional_distribution, mean_distribution)
        print(f"Wilcoxon signed-rank test optimal vs conventional: {p_value_optimal_conventional_wilcoxon}")
        print(f"Wilcoxon signed-rank test optimal vs mean: {p_value_optimal_mean_wilcoxon}")
        print(f"Wilcoxon signed-rank test conventional vs mean: {p_value_conventional_mean_wilcoxon}")

    #TODO combine the following three functions into one
    def plot_LOOCV_YoudenIndex(self):
        self.check_ANOVA_ttest_Wilcoxon(self.optimal_YI_list_ndh, self.conventioanl_YI_list_ndh, self.mean_YI_list_ndh)

        # Set colors for the boxplots
        colors = [thesis_style.get_thesis_colours()['dark_blue'], thesis_style.get_thesis_colours()['light_blue'], thesis_style.get_thesis_colours()['turquoise']]
        mean_markers = dict(marker='D', markerfacecolor=thesis_style.get_thesis_colours()['black'], markersize=5, markeredgewidth=0)
        meadian_markers = dict(color=thesis_style.get_thesis_colours()['black_grey'])

        fig, ax = plt.subplots(figsize=(12, 6))

        box_conventional = ax.boxplot(self.conventioanl_YI_list_ndh, positions=[1], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers)
        box_optimal = ax.boxplot(self.optimal_YI_list_ndh, positions=[2], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers)
        box_mean = ax.boxplot(self.mean_YI_list_ndh, positions=[3], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers)
        box_optimal_Linus = ax.boxplot(self.optimal_YI_list_ndh_Linus, positions=[4], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers)
        box_mean_Linus = ax.boxplot(self.mean_YI_list_ndh_Linus, positions=[5], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers)

        for box in box_conventional['boxes']:
            box.set(facecolor=colors[1])
        for box in box_optimal['boxes']:
            box.set(facecolor=colors[0])
        for box in box_mean['boxes']:
            box.set(facecolor=colors[2])
        for box in box_optimal_Linus['boxes']:
            box.set(facecolor=colors[0])
        for box in box_mean_Linus['boxes']:
            box.set(facecolor=colors[2])

        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xticklabels(['Conventional', 'Personalized', 'Mean optimal', 'Personalized Linus', 'Mean optimal Linus'])

        plt.ylabel('Youden Index')
        plt.title('Leave one Participant out Cross Validation')
        plt.show()

    def plot_LOOCV_AUC(self):
        self.check_ANOVA_ttest_Wilcoxon(self.optimal_AUC_list_ndh, self.conventioanl_AUC_list_ndh, self.mean_AUC_list_ndh)

        # Set colors for the boxplots
        colors = [thesis_style.get_thesis_colours()['dark_blue'], thesis_style.get_thesis_colours()['light_blue'], thesis_style.get_thesis_colours()['orange'], thesis_style.get_thesis_colours()['turquoise']]
        mean_markers = dict(marker='D', markerfacecolor=thesis_style.get_thesis_colours()['black'], markersize=5, markeredgewidth=0)
        meadian_markers = dict(color=thesis_style.get_thesis_colours()['black_grey'])

        fig, ax = plt.subplots(figsize=(12, 6))

        box_conventional = ax.boxplot(self.conventioanl_AUC_list_ndh, positions=[1], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers)
        box_optimal = ax.boxplot(self.optimal_AUC_list_ndh, positions=[2], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers)
        box_mean = ax.boxplot(self.mean_AUC_list_ndh, positions=[3], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers)
        box_optimal_Linus = ax.boxplot(self.optimal_AUC_list_ndh_Linus, positions=[4], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers)
        box_mean_Linus = ax.boxplot(self.mean_AUC_list_ndh_Linus, positions=[5], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers)

        for box in box_conventional['boxes']:
            box.set(facecolor=colors[1])
        for box in box_optimal['boxes']:
            box.set(facecolor=colors[0])
        for box in box_mean['boxes']:
            box.set(facecolor=colors[3])
        for box in box_optimal_Linus['boxes']:
            box.set(facecolor=colors[0])
        for box in box_mean_Linus['boxes']:
            box.set(facecolor=colors[3])

        # AUC is clinically useful (≥0.75) according to [Fan et al., 2006]
        ax.axhline(y=0.75, color=colors[2], linestyle='--')

        # Add a legend manually
        plt.legend([plt.Line2D([0], [0], color=colors[2], linestyle='--')],
                   ['Clinically required performance'],
                   loc='lower left')

        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xticklabels(['Conventional', 'Personalized', 'Mean optimal', 'Personalized Linus', 'Mean optimal Linus'])

        plt.rcParams.update({'font.size': 12})
        plt.ylabel('Area Under the Receiver Operating Characteristic Curve \n(ROC AUC)')
        plt.title('Leave one Participant out Cross Validation')
        plt.show()

    def plot_LOOCV_Accuracy(self):
        self.check_ANOVA_ttest_Wilcoxon(self.optimal_accuracy_list_ndh, self.conventioanl_accuracy_list_ndh, self.mean_accuracy_list_ndh)

        # Set colors for the boxplots
        colors = [thesis_style.get_thesis_colours()['dark_blue'], thesis_style.get_thesis_colours()['light_blue'], thesis_style.get_thesis_colours()['orange'], thesis_style.get_thesis_colours()['turquoise']]
        mean_markers = dict(marker='D', markerfacecolor=thesis_style.get_thesis_colours()['black'], markersize=5, markeredgewidth=0)
        meadian_markers = dict(color=thesis_style.get_thesis_colours()['black_grey'])

        fig, ax = plt.subplots(figsize=(12, 6))

        box_conventional = ax.boxplot(self.conventioanl_accuracy_list_ndh, positions=[1], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers)
        box_optimal = ax.boxplot(self.optimal_accuracy_list_ndh, positions=[2], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers)
        box_mean = ax.boxplot(self.mean_accuracy_list_ndh, positions=[3], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers)
        box_optimal_Linus = ax.boxplot(self.optimal_accuracy_list_ndh_Linus, positions=[4], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers)
        box_mean_Linus = ax.boxplot(self.mean_accuracy_list_ndh_Linus, positions=[5], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers)

        for box in box_conventional['boxes']:
            box.set(facecolor=colors[1])
        for box in box_optimal['boxes']:
            box.set(facecolor=colors[0])
        for box in box_mean['boxes']:
            box.set(facecolor=colors[3])
        for box in box_optimal_Linus['boxes']:
            box.set(facecolor=colors[0])
        for box in box_mean_Linus['boxes']:
            box.set(facecolor=colors[3])

        # Accuracy is clinically useful (≥90%) according to [Lang et al., 2020]
        ax.axhline(y=0.9, color=colors[2], linestyle='--')

        # Add a legend manually
        plt.legend([plt.Line2D([0], [0], color=colors[2], linestyle='--')],
                   ['Clinically required performance'],
                   loc='lower left')
        
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xticklabels(['Conventional', 'Personalized', 'Mean optimal', 'Personalized Linus', 'Mean optimal Linus'])        
        
        plt.ylabel('Accuracy')
        plt.title('Leave one Participant out Cross Validation')
        plt.show()


    def plot_LOOCV_Accuracy_perTask(self):
        plt.rcParams.update({'font.size': 16})
        fig, ax = plt.subplots(figsize=(24, 8))

        # Number of tasks
        num_tasks = len(self.optimal_accuracy_perTask_ndh)
        
        # Prepare the positions for the boxplots
        positions_optimal = np.arange(1, num_tasks * 2, 2)
        positions_conventional = positions_optimal + 0.8

        # Set colors for the boxplots
        colors = [thesis_style.get_thesis_colours()['dark_blue'], thesis_style.get_thesis_colours()['light_blue'], thesis_style.get_thesis_colours()['orange']]
        mean_markers = dict(marker='D', markerfacecolor=thesis_style.get_thesis_colours()['black'], markersize=5, markeredgewidth=0)
        meadian_markers = dict(color=thesis_style.get_thesis_colours()['black_grey'])
        
        # Plot the boxplots for optimal and conventional accuracies with patch_artist=True to allow color filling
        box_optimal = ax.boxplot(self.optimal_accuracy_perTask_ndh.values(), positions=positions_optimal, widths=0.6, showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers)
        box_conventional = ax.boxplot(self.conventional_accuracy_perTask_ndh.values(), positions=positions_conventional, widths=0.6, showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers)
        for box in box_optimal['boxes']:
            box.set(facecolor=colors[0])
        for box in box_conventional['boxes']:
            box.set(facecolor=colors[1])

        # Accuracy is clinically useful (≥90%) according to [Lang et al., 2020]
        ax.axhline(y=0.9, color=colors[2], linestyle='--')
        
        # Adjust x-tick labels to be in between the grouped boxplots
        mid_positions = (positions_optimal + positions_conventional) / 2
        ax.set_xticks(mid_positions)
        ax.set_xticklabels(self.optimal_accuracy_perTask_ndh.keys())
        plt.xticks(rotation=45, ha='right')
        
        plt.ylabel('Accuracy')
        plt.title('Leave one Participant out Cross Validation per Task')
        
        # Add a legend manually
        plt.legend([plt.Line2D([0], [0], color=colors[0], lw=10),
                    plt.Line2D([0], [0], color=colors[1], lw=10),
                    plt.Line2D([0], [0], color=colors[2], linestyle='--')],
                ['Personalized', 'Conventional', 'Clinically required performance'], loc='lower left')

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