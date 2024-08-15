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
        task_mask_25Hz = []
        for path in json_files:
            dict_1Hz = extract_fields_from_json_files([path], ['GT_mask_NDH_1Hz', 'GT_mask_DH_1Hz', 'counts_for_GMAC_ndh_1Hz', 'counts_for_GMAC_dh_1Hz', 'task_mask_25Hz', 'pitch_for_GMAC_ndh_1Hz', 'pitch_for_GMAC_dh_1Hz'])
            task_mask_25Hz.append(dict_1Hz['task_mask_25Hz'])
            GT_NDH_1Hz.append(dict_1Hz['GT_mask_NDH_1Hz'])
            GT_DH_1Hz.append(dict_1Hz['GT_mask_DH_1Hz'])
            counts_NDH_1Hz.append(dict_1Hz['counts_for_GMAC_ndh_1Hz'])
            counts_DH_1Hz.append(dict_1Hz['counts_for_GMAC_dh_1Hz'])
            elevation_NDH_1Hz.append(dict_1Hz['pitch_for_GMAC_ndh_1Hz'])
            elevation_DH_1Hz.append(dict_1Hz['pitch_for_GMAC_dh_1Hz'])
        task_mask = {'task_mask_25Hz': task_mask_25Hz}
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
        FMA = [participant['FMA_UE'] for participant in X_test]
        count_predict = count_threshold_model.predict_polynomial(FMA, 2)
        elevation_predict = elevation_threshold_model.predict_polynomial(FMA, 2)

        print("Validation on participant ", X_test[0]['participant_id'])
        print("Polynomial Predictions of personal COUNT threshold: ", count_predict)
        print("Polynomial Predictions of personal ELEVATION threshold: ", elevation_predict)

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

        AUC_analisys(test_gt, gmac_prediction.flatten())
        accuracy_analisys(accuracy)

        return youden_index, accuracy

    def LOOCV_complete(self):

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

        for train_index, test_index in looCV.split(X, y, groups=group_IDs):
            
            X_train, y_train = [X[i] for i in train_index], [y[i] for i in train_index]
            X_test, y_test = [X[i] for i in test_index], [y[i] for i in test_index]

            regression_model_count_ndh, regression_model_elevation_ndh = self.get_threshold_model(X_train)
            #threshold_model_dh = self.get_threshold_model()

            personalized_count_threshold_ndh, personalized_elevation_threshold_ndh = self.retreive_personalized_thresholds(X_test, regression_model_count_ndh, regression_model_elevation_ndh)
            #personalized_thresholds_dh = self.retreive_personalized_thresholds(threshold_model_dh)

            youden_index_optimized, accuracy_optimized = self.calculate_GMAC_classification_performance(X_test, y_test, personalized_count_threshold_ndh, personalized_elevation_threshold_ndh)
            youden_index_conventional, accuracy_conventional = self.calculate_GMAC_classification_performance(X_test, y_test, 0, 30)
            #dh_performance = self.calculate_GMAC_classification_performance(personalized_thresholds_dh)

            self.optimal_YI_list.append(youden_index_optimized)
            self.conventioanl_YI_list.append(youden_index_conventional)
            self.optimal_accuracy_list.append(accuracy_optimized)
            self.conventioanl_accuracy_list.append(accuracy_conventional)

    def LOOCV_perTask(self):
        pass

    def plot_LOOCV_YoudenIndex(self):
        plt.boxplot([self.optimal_YI_list, self.conventioanl_YI_list], showmeans=True)
        plt.ylabel('Youden Index')
        plt.title('Leave one Participant out Cross Validation')
        plt.xticks([1, 2], ['Personalized', 'Conventional'])
        plt.show()

    def plot_LOOCV_Accuracy(self):
        plt.boxplot([self.optimal_accuracy_list, self.conventioanl_accuracy_list], showmeans=True)
        plt.ylabel('Accuracy')
        plt.title('Leave one Participant out Cross Validation')
        plt.xticks([1, 2], ['Personalized', 'Conventional'])
        plt.show()