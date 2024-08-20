import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from utilities import *
from individual_analysis_gmac_function import get_prediction_gmac, AUC_analisys, accuracy_analisys
from functions.statistics import RegressionModel


#TODO fix naming with correct number of Hz for counts pitch and GT, ...
class LOOCV_performance:
    def __init__(self, json_files):

        participant_data = extract_fields_from_json_files(json_files, ['optimal_GMAC_NDH', 'optimal_GMAC_DH', 'ARAT_score', 'FMA-UE_score', 'participant_id'])

        self.PARTICIPANT_ID = participant_data['participant_id']
        self.ARAT = participant_data['ARAT_score']
        self.FMA_UE = participant_data['FMA-UE_score']

        optimal_thresholds = participant_data['optimal_GMAC_NDH']
        self.COUNT_THRESHOLD_NDH = optimal_thresholds[:,0]
        self.PITCH_THRESHOLD_NDH = optimal_thresholds[:,1]
        optimal_thresholds = participant_data['optimal_GMAC_DH']
        self.COUNT_THRESHOLD_DH = optimal_thresholds[:,0]
        self.PITCH_THRESHOLD_DH = optimal_thresholds[:,1]

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

    def get_threshold_model(self, X_train):
        FMA_array = np.array([participant['FMA_UE'] for participant in X_train])
        count_threshold_array = np.array([participant['COUNT_THRESHOLD_NDH'] for participant in X_train])
        elevation_threshold_array = np.array([participant['PITCH_THRESHOLD_NDH'] for participant in X_train])

        regression_model_count = RegressionModel(FMA_array, count_threshold_array)
        regression_model_count.fit_polynomial_regression(2)

        regression_model_elevation = RegressionModel(FMA_array, elevation_threshold_array)
        regression_model_elevation.fit_polynomial_regression(2)

        return regression_model_count, regression_model_elevation


    def retreive_personalized_thresholds(self, X_test, count_threshold_model, elevation_threshold_model):
        assert len(X_test) == 1, "X_test should contain only one participant"
        FMA = X_test[0]['FMA_UE']
        count_predict = count_threshold_model.predict_polynomial(FMA, 2)
        elevation_predict = elevation_threshold_model.predict_polynomial(FMA, 2)

        print("Validation on participant ", X_test[0]['participant_id'])
        print("Polynomial Predictions of personal COUNT threshold: ", count_predict, ". Ground truth individual optimal threshold NDH: ", X_test[0]['COUNT_THRESHOLD_NDH'])
        print("Polynomial Predictions of personal ELEVATION threshold: ", elevation_predict, ". Ground truth individual optimal threshold NDH: ", X_test[0]['PITCH_THRESHOLD_NDH'])

        return count_predict, elevation_predict

    def calculate_GMAC_classification_performance(self, X_test, y_test, personalized_count_threshold, personalized_elevation_threshold):
        assert len(X_test) == 1, "X_test should contain only one participant"
        assert len(y_test) == 1, "y_test should contain only one participant"
        test_gt = y_test[0].flatten()

        counts_array = np.array(X_test[0]['counts_NDH_1Hz'])
        elevation_array = np.array(X_test[0]['elevation_NDH_1Hz'])

        gmac_prediction = get_prediction_gmac(counts_array, elevation_array, count_threshold=personalized_count_threshold, functional_space=personalized_elevation_threshold, decision_mode='Subash')
        gmac_prediction = gmac_prediction.astype(int)

        tn, fp, fn, tp = confusion_matrix(test_gt, gmac_prediction.flatten()).ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        youden_index = sensitivity + specificity - 1

        auc = AUC_analisys(test_gt, gmac_prediction.flatten())
        accuracy_analisys(accuracy)

        return youden_index, accuracy, auc
    
    def calculate_GMAC_classification_performance_perTask(self, task_dict, personalized_count_threshold, personalized_elevation_threshold):
        YI_per_task = {} #doesn't make sense if only one label is present the gt (functional)
        accuracy_per_task = {}
        for task_of_interest, task_data in task_dict.items():
            count_for_task = np.array(task_data['count'])
            pitch_for_task = np.array(task_data['elevation'])
            gt_for_task = task_data['gt'].flatten()

            gmac_prediction = get_prediction_gmac(count_for_task, pitch_for_task, count_threshold=personalized_count_threshold, functional_space=personalized_elevation_threshold, decision_mode='Subash')
            gmac_prediction = gmac_prediction.astype(int)

            tn, fp, fn, tp = confusion_matrix(gt_for_task, gmac_prediction, labels=[1, 0]).ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            #sensitivity = tp / (tp + fn)
            #specificity = tn / (tn + fp)
            #youden_index = sensitivity + specificity - 1

            #YI_per_task[task_of_interest] = youden_index
            accuracy_per_task[task_of_interest] = accuracy

        return YI_per_task, accuracy_per_task


    def LOOCV_complete(self, perTask=True):

        participant_dicts = []
        for i, participant_id in enumerate(self.PARTICIPANT_ID):
            participant_dict = {
                'participant_id': participant_id,
                'ARAT': self.ARAT[i],
                'FMA_UE': self.FMA_UE[i],
                'COUNT_THRESHOLD_NDH': self.COUNT_THRESHOLD_NDH[i],
                'PITCH_THRESHOLD_NDH': self.PITCH_THRESHOLD_NDH[i],
                'counts_NDH_1Hz': self.count_data['counts_NDH_1Hz'][i],
                'elevation_NDH_1Hz': self.pitch_data['elevation_NDH_1Hz'][i],
                'task_mask_ndh_1Hz': self.task_mask['task_mask_ndh_1Hz'][i]
            }
            participant_dicts.append(participant_dict)

        looCV = LeaveOneGroupOut()
        X = participant_dicts
        y = self.gt_functional['GT_mask_NDH_1Hz']
        group_IDs = self.PARTICIPANT_ID

        self.optimal_YI_list = []
        self.conventioanl_YI_list = []
        self.optimal_accuracy_list = []
        self.conventioanl_accuracy_list = []
        self.optimal_AUC_list = []
        self.conventioanl_AUC_list = []

        self.optimal_accuracy_perTask = {}
        self.conventional_accuracy_perTask = {}

        for train_index, test_index in looCV.split(X, y, groups=group_IDs):
            
            X_train, _ = [X[i] for i in train_index], [y[i] for i in train_index]
            X_test, y_test = [X[i] for i in test_index], [y[i] for i in test_index]

            regression_model_count_ndh, regression_model_elevation_ndh = self.get_threshold_model(X_train)
            #threshold_model_dh = self.get_threshold_model()

            personalized_count_threshold_ndh, personalized_elevation_threshold_ndh = self.retreive_personalized_thresholds(X_test, regression_model_count_ndh, regression_model_elevation_ndh)
            #personalized_thresholds_dh = self.retreive_personalized_thresholds(threshold_model_dh)

            youden_index_optimized, accuracy_optimized, auc_optimized = self.calculate_GMAC_classification_performance(X_test, y_test, personalized_count_threshold_ndh, personalized_elevation_threshold_ndh)
            youden_index_conventional, accuracy_conventional, auc_conventional = self.calculate_GMAC_classification_performance(X_test, y_test, 0, 30)
            #dh_performance = self.calculate_GMAC_classification_performance(personalized_thresholds_dh)

            self.optimal_YI_list.append(youden_index_optimized)
            self.conventioanl_YI_list.append(youden_index_conventional)
            self.optimal_accuracy_list.append(accuracy_optimized)
            self.conventioanl_accuracy_list.append(accuracy_conventional)
            self.optimal_AUC_list.append(auc_optimized)
            self.conventioanl_AUC_list.append(auc_conventional)
            #TODO plot predicted and ground truth personalized count threshold and elevation threshold

            if perTask:
                self.LOOCV_perTask(X_test, y_test, personalized_count_threshold_ndh, personalized_elevation_threshold_ndh)

    def LOOCV_perTask(self, X_test, y_test, personalized_count_threshold_ndh, personalized_elevation_threshold_ndh):
        protocol_tasks = ['open_bottle_and_pour_glass', 'drink', 'fold_rags_towels', 'sort_documents', 'brooming', 'putting_on_and_off_coat', 'keyboard_typing', 'stapling', 'walking', 
                          'open_and_close_door', 'resting', 'other', 'wipe_table','light_switch']
        task_dict = {}
        for task_of_interest in protocol_tasks:
            count_for_task = extract_all_values_with_label(X_test[0]['counts_NDH_1Hz'], X_test[0]['task_mask_ndh_1Hz'], task_of_interest)
            pitch_for_task = extract_all_values_with_label(X_test[0]['elevation_NDH_1Hz'], X_test[0]['task_mask_ndh_1Hz'], task_of_interest)
            gt_for_task = extract_all_values_with_label(y_test[0], X_test[0]['task_mask_ndh_1Hz'], task_of_interest)
            task_dict[task_of_interest] = {'count': count_for_task, 'elevation': pitch_for_task, 'gt': gt_for_task}

        _, accuracy_per_task_optimized = self.calculate_GMAC_classification_performance_perTask(task_dict, personalized_count_threshold_ndh, personalized_elevation_threshold_ndh)
        _, accuracy_per_task_conventional = self.calculate_GMAC_classification_performance_perTask(task_dict, 0, 30)

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
        

    def plot_LOOCV_YoudenIndex(self):
        # Set colors for the boxplots
        colors = [thesis_style.get_thesis_colours()['dark_blue'], thesis_style.get_thesis_colours()['light_blue']]
        mean_markers = dict(marker='D', markerfacecolor=thesis_style.get_thesis_colours()['black'], markersize=5, markeredgewidth=0)
        meadian_markers = dict(color=thesis_style.get_thesis_colours()['black_grey'])

        fig, ax = plt.subplots()

        box_optimal = ax.boxplot(self.optimal_YI_list, positions=[1], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers)
        box_conventional = ax.boxplot(self.conventioanl_YI_list, positions=[2], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers)

        for box in box_optimal['boxes']:
            box.set(facecolor=colors[0])
        for box in box_conventional['boxes']:
            box.set(facecolor=colors[1])

        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Personalized', 'Conventional'])

        plt.ylabel('Youden Index')
        plt.title('Leave one Participant out Cross Validation')
        plt.show()

    def plot_LOOCV_AUC(self):
        # Set colors for the boxplots
        colors = [thesis_style.get_thesis_colours()['dark_blue'], thesis_style.get_thesis_colours()['light_blue'], thesis_style.get_thesis_colours()['orange']]
        mean_markers = dict(marker='D', markerfacecolor=thesis_style.get_thesis_colours()['black'], markersize=5, markeredgewidth=0)
        meadian_markers = dict(color=thesis_style.get_thesis_colours()['black_grey'])

        fig, ax = plt.subplots()

        box_optimal = ax.boxplot(self.optimal_AUC_list, positions=[1], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers)
        box_conventional = ax.boxplot(self.conventioanl_AUC_list, positions=[2], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers)

        for box in box_optimal['boxes']:
            box.set(facecolor=colors[0])
        for box in box_conventional['boxes']:
            box.set(facecolor=colors[1])

        # AUC is clinically useful (≥0.75) according to [Fan et al., 2006]
        ax.axhline(y=0.75, color=colors[2], linestyle='--')

        # Add a legend manually
        plt.legend([plt.Line2D([0], [0], color=colors[2], linestyle='--')],
                   ['Clinically required performance'],
                   loc='lower left')

        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Personalized', 'Conventional'])

        plt.ylabel('Area Under the Receiver Operating Characteristic Curve \n(ROC AUC)')
        plt.title('Leave one Participant out Cross Validation')
        plt.show()

    def plot_LOOCV_Accuracy(self):
        # Set colors for the boxplots
        colors = [thesis_style.get_thesis_colours()['dark_blue'], thesis_style.get_thesis_colours()['light_blue'], thesis_style.get_thesis_colours()['orange']]
        mean_markers = dict(marker='D', markerfacecolor=thesis_style.get_thesis_colours()['black'], markersize=5, markeredgewidth=0)
        meadian_markers = dict(color=thesis_style.get_thesis_colours()['black_grey'])

        fig, ax = plt.subplots()

        box_optimal = ax.boxplot(self.optimal_accuracy_list, positions=[1], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers)
        box_conventional = ax.boxplot(self.conventioanl_accuracy_list, positions=[2], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers)

        for box in box_optimal['boxes']:
            box.set(facecolor=colors[0])
        for box in box_conventional['boxes']:
            box.set(facecolor=colors[1])

        # Accuracy is clinically useful (≥90%) according to [Lang et al., 2020]
        ax.axhline(y=0.9, color=colors[2], linestyle='--')

        # Add a legend manually
        plt.legend([plt.Line2D([0], [0], color=colors[2], linestyle='--')],
                   ['Clinically required performance'],
                   loc='lower left')
        
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Personalized', 'Conventional'])        
        
        plt.ylabel('Accuracy')
        plt.title('Leave one Participant out Cross Validation')
        plt.show()

    def plot_LOOCV_Accuracy_perTask(self):
        plt.rcParams.update({'font.size': 16})
        fig, ax = plt.subplots(figsize=(24, 8))

        # Number of tasks
        num_tasks = len(self.optimal_accuracy_perTask)
        
        # Prepare the positions for the boxplots
        positions_optimal = np.arange(1, num_tasks * 2, 2)
        positions_conventional = positions_optimal + 0.8

        # Set colors for the boxplots
        colors = [thesis_style.get_thesis_colours()['dark_blue'], thesis_style.get_thesis_colours()['light_blue'], thesis_style.get_thesis_colours()['orange']]
        mean_markers = dict(marker='D', markerfacecolor=thesis_style.get_thesis_colours()['black'], markersize=5, markeredgewidth=0)
        meadian_markers = dict(color=thesis_style.get_thesis_colours()['black_grey'])
        
        # Plot the boxplots for optimal and conventional accuracies with patch_artist=True to allow color filling
        box_optimal = ax.boxplot(self.optimal_accuracy_perTask.values(), positions=positions_optimal, widths=0.6, showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers)
        box_conventional = ax.boxplot(self.conventional_accuracy_perTask.values(), positions=positions_conventional, widths=0.6, showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers)
        for box in box_optimal['boxes']:
            box.set(facecolor=colors[0])
        for box in box_conventional['boxes']:
            box.set(facecolor=colors[1])

        # Accuracy is clinically useful (≥90%) according to [Lang et al., 2020]
        ax.axhline(y=0.9, color=colors[2], linestyle='--')
        
        # Adjust x-tick labels to be in between the grouped boxplots
        mid_positions = (positions_optimal + positions_conventional) / 2
        ax.set_xticks(mid_positions)
        ax.set_xticklabels(self.optimal_accuracy_perTask.keys())
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