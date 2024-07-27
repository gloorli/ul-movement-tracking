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
        #primitives = extract_fields_from_json_files(s_json_files, ['primitive_mask_LW_25Hz', 'primitive_mask_RW_25Hz'])

        self.participantIDs = result['participant_id']
        self.affected_arms = result['affected_hand']
        self.ARATs = result['ARAT_score']
        self.FMA_UEs = result['FMA-UE_score']
        self.primitive_LWs = primitives['primitive_mask_LW_25Hz']
        self.primitive_RWs = primitives['primitive_mask_RW_25Hz']
        self.label_to_int = {'functional_movement': 1, 'non_functional_movement': 0, 'reach': 2, 'reposition': 3, 'transport': 4, 'gesture': 5, 'idle': 6, 'stabilization': 7, 'arm_not_visible': 999}

    def count_primitives(self):
        self.primitive_amount_LW = []
        self.primitive_amount_RW = []
        for i in range(len(self.participantIDs)):
            unique_values_LW, amount_LW = np.unique(self.primitive_LWs[i], return_counts=True)
            LW_dict = dict(zip(unique_values_LW, amount_LW))
            unique_values_RW, amount_RW = np.unique(self.primitive_RWs[i], return_counts=True)
            RW_dict = dict(zip(unique_values_RW, amount_RW))
        
            amount_LW_full = [None]*len(self.label_to_int)
            for key, value in LW_dict.items():
                amount_LW_full[list(self.label_to_int.values()).index(key)] = value
            amount_LW_full.insert(0, self.participantIDs[i])
            self.primitive_amount_LW.append(amount_LW_full)

            amount_RW_full = [None]*len(self.label_to_int)
            for key, value in RW_dict.items():
                amount_RW_full[list(self.label_to_int.values()).index(key)] = value
            amount_RW_full.insert(0, self.participantIDs[i])
            self.primitive_amount_RW.append(amount_RW_full)

    def plot_primitive_distribution(self, side='LW'):
        labels = list(self.label_to_int.keys())
        labels.insert(0, 'participantID')
        if side == 'LW':
            data = self.primitive_amount_LW
        elif side == 'RW':
            data = self.primitive_amount_RW
        else:
            raise ValueError('side must be either "LW" or "RW"')
        df = pd.DataFrame(data, columns=labels)
        df.plot(x='participantID', kind='bar', stacked=True, title='Primitives '+side, legend=True)
        plt.ylabel('Frames')
        plt.show()



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