from utilities import *

class PrimitiveDistribution:
    def __init__(self, initial_path = '../data/CreateStudy'):
        
        s_json_files = get_json_paths(initial_path, 'S')
        result = extract_fields_from_json_files(s_json_files, ['participant_id', 'affected_hand', 'ARAT_score', 'FMA-UE_score'])
        primitives_LW = []
        primitives_RW = []
        for path in s_json_files:
            dict = extract_fields_from_json_files([path], ['primitive_mask_LW_25Hz', 'primitive_mask_RW_25Hz'])
            primitives_LW.append(dict['primitive_mask_LW_25Hz'])
            primitives_RW.append(dict['primitive_mask_RW_25Hz'])
        primitives = {'primitive_mask_LW_25Hz': primitives_LW, 'primitive_mask_RW_25Hz': primitives_RW}

        self.participantIDs = result['participant_id']
        self.affected_arms = result['affected_hand']
        self.ARATs = result['ARAT_score']
        self.FMA_UEs = result['FMA-UE_score']
        self.primitive_LWs = primitives['primitive_mask_LW_25Hz']
        self.primitive_RWs = primitives['primitive_mask_RW_25Hz']
        self.label_to_int = {'functional_movement': 1, 'non_functional_movement': 0, 'reach': 2, 'reposition': 3, 'transport': 4, 'gesture': 5, 'idle': 6, 'stabilization': 7, 'arm_not_visible': 999}

    def count_primitives_per_participant(self, primitives):
        unique_values, amount = np.unique(primitives, return_counts=True)
        return dict(zip(unique_values, amount))
    
    def fill_participant_primitive_list(self, dict, i):
        amount_full = [None]*len(self.label_to_int)
        for key, value in dict.items():
            amount_full[list(self.label_to_int.values()).index(key)] = value
        amount_full.insert(0, self.participantIDs[i])
        return amount_full

    def count_primitives_NDHDH(self):
        self.primitive_amount_NDH = []
        self.primitive_amount_DH = []
        self.primitives_NDHs, self.primitive_DHs = from_LWRW_to_NDHDH(self.affected_arms, {'primitive_mask_LW_25Hz': self.primitive_LWs, 'primitive_mask_RW_25Hz': self.primitive_RWs})
        for i in range(len(self.participantIDs)):
            NDH_dict = self.count_primitives_per_participant(self.primitives_NDHs[i])
            DH_dict = self.count_primitives_per_participant(self.primitive_DHs[i])
            self.primitive_amount_NDH.append(self.fill_participant_primitive_list(NDH_dict, i))
            self.primitive_amount_DH.append(self.fill_participant_primitive_list(DH_dict, i))

    def count_primitives_LWRW(self):
        self.primitive_amount_LW = []
        self.primitive_amount_RW = []
        for i in range(len(self.participantIDs)):
            LW_dict = self.count_primitives_per_participant(self.primitive_LWs[i])
            RW_dict = self.count_primitives_per_participant(self.primitive_RWs[i])
            self.primitive_amount_LW.append(self.fill_participant_primitive_list(LW_dict, i))
            self.primitive_amount_RW.append(self.fill_participant_primitive_list(RW_dict, i))

    def primitivecounts_to_percentage(self, df_percentage):
        '''
        Converts the counts of primitives to percentages of primitives.
        Parameters:
        - df_percentage (DataFrame): A DataFrame containing the counts of primitives for each participant.
        Returns:
        - df_percentage (DataFrame): A DataFrame containing the percentages of primitives for each participant.
        '''
        df_percentage.set_index('participantID', inplace=True)
        total_frames = df_percentage.sum(axis=1)
        df_percentage = df_percentage.div(total_frames, axis=0) * 100
        df_percentage.reset_index(inplace=True)
        return df_percentage
    
    def order_df_by_FMA(self, df):
        """
        Orders the given DataFrame by the 'FMA_UE' column and returns the sorted DataFrame along with the 'participantID',
        'FMA_UE', and 'ARAT' columns as separate variables. If the 'FMA_UE' column contains the same values for multiple rows, the rows are sorted by the 'ARAT' column.
        Parameters:
            df (pandas.DataFrame): The DataFrame to be sorted.
        Returns:
            tuple: A tuple containing the sorted DataFrame, 'participantID' labels, 'FMA_UE' labels, and 'ARAT' labels.
        """
        df['FMA_UE'] = self.FMA_UEs
        df['ARAT'] = self.ARATs
        df.sort_values(['FMA_UE', 'ARAT'], inplace=True)
        ID_labels = df['participantID']
        FMA_labels = df.pop('FMA_UE')
        ARAT_labels = df.pop('ARAT')
        return df, ID_labels, FMA_labels, ARAT_labels

    def plot_primitive_distribution(self, side='LW'):
        labels = list(self.label_to_int.keys())
        label_for_colors = labels[2:]
        labels.insert(0, 'participantID')
        if side == 'LW':
            data = self.primitive_amount_LW
        elif side == 'RW':
            data = self.primitive_amount_RW
        elif side == 'NDH':
            data = self.primitive_amount_NDH
        elif side == 'DH':
            data = self.primitive_amount_DH
        else:
            raise ValueError('side must be either "NDH", "DH", "LW" or "RW"')
        df = pd.DataFrame(data, columns=labels)
        df.drop(columns=['functional_movement', 'non_functional_movement', 'arm_not_visible'], inplace=True)
        df_percentage = self.primitivecounts_to_percentage(df.copy())
        df_percentage_ordered, ID_label, FMA_label, ARAT_label = self.order_df_by_FMA(df_percentage.copy())
        df_percentage_ordered.plot(x='participantID', kind='bar', stacked=True, title='Primitive Distribution '+side, legend=True,
                color=[thesis_style.get_label_colours()[key] for key in label_for_colors])
        plt.ylabel('percentage of protocol time')
        plt.xlabel('')
        plt.xticks(range(len(self.participantIDs)), [f"{id}\nFMA: {fma}\nARAT: {arat}" for id, fma, arat in zip(ID_label, FMA_label, ARAT_label)], rotation=0, fontsize=8)
        plt.yticks(range(0, 101, 25), [f"{i}%" for i in range(0, 101, 25)])
        plt.tight_layout()
        plt.legend(loc='center right', bbox_to_anchor=(1.3, 0.5), reverse=True)
        plt.show()

    def plot_primitive_distribution_per_group(self):
        # FMA-UE cut-off scores: mild (43–66), moderate (29–42), and severe (0–18) (Woytowicz et al., 2017).
        pass


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_primitive_mask(primitive_mask, participantID, int_to_label):
    unique_values, amount = np.unique(primitive_mask, return_counts=True)
    labels = [int_to_label.get(value, 'Unknown') for value in unique_values]
    data = amount.tolist()
    data.insert(0, participantID)
    labels.insert(0, 'participantID')
    df = pd.DataFrame([data], columns=labels)
    df.plot(x='participantID', kind='bar', stacked=True, title='Primitives', legend=True)
    plt.ylabel('Frames')
    plt.show()