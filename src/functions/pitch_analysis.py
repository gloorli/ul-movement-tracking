import numpy as np
from utilities import *


class PitchPerPrimitive:
    def __init__(self, label_to_int, initial_path = '../data/CreateStudy'):
        
        self.label_to_int = label_to_int.copy()
        self.pitch_per_primitive_ndh = self.label_to_int.copy()
        self.pitch_per_primitive_dh = self.label_to_int.copy()
        self.mean_pitch_per_primitive = self.label_to_int.copy()
        self.pitch_per_task_ndh = {}
        self.pitch_per_task_dh = {}
        self.count_per_task_ndh = {}
        self.count_per_task_dh = {}

        s_json_files = get_json_paths(initial_path, 'S')

        result = extract_fields_from_json_files(s_json_files, ['participant_id', 'affected_hand', 'ARAT_score', 'FMA-UE_score'])
        self.participant_id = result['participant_id']
        self.affected_hand = result['affected_hand']
        self.ARAT_score = result['ARAT_score']
        self.FMA_UE_score = result['FMA-UE_score']

        primitives_LW = []
        primitives_RW = []
        pitch_NDH_25Hz = []
        pitch_DH_25Hz = []
        GT_NDH_25Hz = []
        GT_DH_25Hz = []
        task_NDH_1Hz, count_NDH_1Hz, elevation_NDH_1Hz = [], [], []
        task_DH_1Hz, count_DH_1Hz, elevation_DH_1Hz = [], [], []
        for path in s_json_files:
            gt_dict = extract_fields_from_json_files([path], ['primitive_mask_LW_25Hz', 'primitive_mask_RW_25Hz', 'GT_mask_NDH_25Hz', 'GT_mask_DH_25Hz'])
            pitch_dict_50Hz = extract_fields_from_json_files([path], ['elevation_NDH_50Hz', 'elevation_DH_50Hz'])
            task_dict_1Hz = extract_fields_from_json_files([path], ['task_mask_for_GMAC_NDH_1Hz', 'task_mask_for_GMAC_DH_1Hz', 
                                                                    'counts_for_GMAC_ndh_1Hz', 'counts_for_GMAC_dh_1Hz', 'pitch_for_GMAC_ndh_1Hz', 'pitch_for_GMAC_dh_1Hz'])
            primitives_LW.append(gt_dict['primitive_mask_LW_25Hz'])
            primitives_RW.append(gt_dict['primitive_mask_RW_25Hz'])
            GT_NDH_25Hz.append(gt_dict['GT_mask_NDH_25Hz'])
            GT_DH_25Hz.append(gt_dict['GT_mask_DH_25Hz'])
            pitch_NDH_25Hz.append(np.average(pitch_dict_50Hz['elevation_NDH_50Hz'].reshape(-1, 2), axis=1))
            pitch_DH_25Hz.append(np.average(pitch_dict_50Hz['elevation_DH_50Hz'].reshape(-1, 2), axis=1))
            task_NDH_1Hz.append(task_dict_1Hz['task_mask_for_GMAC_NDH_1Hz'])
            task_DH_1Hz.append(task_dict_1Hz['task_mask_for_GMAC_DH_1Hz'])
            count_NDH_1Hz.append(task_dict_1Hz['counts_for_GMAC_ndh_1Hz'])
            count_DH_1Hz.append(task_dict_1Hz['counts_for_GMAC_dh_1Hz'])
            elevation_NDH_1Hz.append(task_dict_1Hz['pitch_for_GMAC_ndh_1Hz'])
            elevation_DH_1Hz.append(task_dict_1Hz['pitch_for_GMAC_dh_1Hz'])
        primitives = {'primitive_mask_LW_25Hz': primitives_LW, 'primitive_mask_RW_25Hz': primitives_RW}
        gt_functional = {'GT_mask_NDH_25Hz': GT_NDH_25Hz, 'GT_mask_DH_25Hz': GT_DH_25Hz}
        pitch_data = {'pitch_NDH_25Hz': pitch_NDH_25Hz, 'pitch_DH_25Hz': pitch_DH_25Hz}
        task_1Hz = {'task_NDH_1Hz': task_NDH_1Hz, 'task_DH_1Hz': task_DH_1Hz}
        count_elevation_1Hz = {'count_NDH_1Hz': count_NDH_1Hz, 'count_DH_1Hz': count_DH_1Hz, 'elevation_NDH_1Hz': elevation_NDH_1Hz, 'elevation_DH_1Hz': elevation_DH_1Hz}
        self.primitives = primitives
        self.gt_functional = gt_functional
        self.pitch_data = pitch_data
        self.task_1Hz = task_1Hz
        self.count_elevation_1Hz = count_elevation_1Hz

    def get_pitch_per_primitive(self):
        """
        Get the average pitch per primitive.
        """
        pitch_ndh = self.pitch_data['pitch_NDH_25Hz']
        pitch_dh = self.pitch_data['pitch_DH_25Hz']
        primitives_ndh, primitives_dh = from_LWRW_to_NDHDH(self.affected_hand, self.primitives)

        pitch_ndh = np.concatenate(pitch_ndh, axis=None)
        pitch_dh = np.concatenate(pitch_dh, axis=None)
        combined_pitch = np.concatenate((pitch_ndh, pitch_dh), axis=None)
        
        primitives_ndh = np.concatenate(primitives_ndh, axis=None)
        primitives_dh = np.concatenate(primitives_dh, axis=None)
        combined_primitives = np.concatenate((primitives_ndh, primitives_dh), axis=None)    

        for key, value in self.label_to_int.items():
            mean_pitch = np.mean(extract_all_values_with_label(combined_pitch, combined_primitives, value))
            self.mean_pitch_per_primitive[key] = mean_pitch

        return self.mean_pitch_per_primitive
    
    def get_pitch_per_functional(self):
        """
        Average pitch for functional and non functional periods.
        """
        mean_pitch_functional = np.mean([self.mean_pitch_per_primitive['reach'], self.mean_pitch_per_primitive['reposition'], self.mean_pitch_per_primitive['transport'], self.mean_pitch_per_primitive['gesture']])
        mean_pitch_non_functional = np.mean([self.mean_pitch_per_primitive['idle'], self.mean_pitch_per_primitive['stabilization']])

        self.mean_pitch_per_primitive['functional_movement'] = mean_pitch_functional
        self.mean_pitch_per_primitive['non_functional_movement'] = mean_pitch_non_functional

        return self.mean_pitch_per_primitive
    
    def get_pitch_per_primitive_over_all_participants(self):
        #TODO clean up (remove duplicate code in function get_pitch_per_primitive(self))
        pitch_ndh = self.pitch_data['pitch_NDH_25Hz']
        pitch_dh = self.pitch_data['pitch_DH_25Hz']
        primitives_ndh, primitives_dh = from_LWRW_to_NDHDH(self.affected_hand, self.primitives)
        gt_functional_ndh = self.gt_functional['GT_mask_NDH_25Hz']
        gt_functional_dh = self.gt_functional['GT_mask_DH_25Hz']

        pitch_ndh = np.concatenate(pitch_ndh, axis=None)
        pitch_dh = np.concatenate(pitch_dh, axis=None)
        
        primitives_ndh = np.concatenate(primitives_ndh, axis=None)
        primitives_dh = np.concatenate(primitives_dh, axis=None)

        gt_functional_ndh = np.concatenate(gt_functional_ndh, axis=None)
        gt_functional_dh = np.concatenate(gt_functional_dh, axis=None)

        for key, value in self.label_to_int.items():
            if key == 'functional_movement' or key == 'non_functional_movement':
                pitches_ndh = extract_all_values_with_label(pitch_ndh, gt_functional_ndh, value)
                pitches_nd = extract_all_values_with_label(pitch_dh, gt_functional_dh, value)
                self.pitch_per_primitive_ndh[key], self.pitch_per_primitive_dh[key] = pitches_ndh, pitches_nd
                continue
            pitches_ndh = extract_all_values_with_label(pitch_ndh, primitives_ndh, value)
            pitches_nd = extract_all_values_with_label(pitch_dh, primitives_dh, value)
            self.pitch_per_primitive_ndh[key], self.pitch_per_primitive_dh[key] = pitches_ndh, pitches_nd

    def get_pitch_per_task(self, protocol_tasks):
        """
        Get the pitch per task.
        """
        pitch_ndh = self.count_elevation_1Hz['elevation_NDH_1Hz']
        pitch_dh = self.count_elevation_1Hz['elevation_DH_1Hz']
        task_ndh = self.task_1Hz['task_NDH_1Hz']
        task_dh = self.task_1Hz['task_DH_1Hz']

        pitch_ndh = np.concatenate(pitch_ndh, axis=None)
        pitch_dh = np.concatenate(pitch_dh, axis=None)
        task_ndh = np.concatenate(task_ndh, axis=None)
        task_dh = np.concatenate(task_dh, axis=None)

        for task in protocol_tasks:
            pitches_ndh = extract_all_values_with_label(pitch_ndh, task_ndh, task)
            pitches_nd = extract_all_values_with_label(pitch_dh, task_dh, task)
            self.pitch_per_task_ndh[task], self.pitch_per_task_dh[task] = pitches_ndh, pitches_nd

    def get_count_per_task(self, protocol_tasks):
        """
        Get the count per task.
        """
        count_ndh = self.count_elevation_1Hz['count_NDH_1Hz']
        count_dh = self.count_elevation_1Hz['count_DH_1Hz']
        task_ndh = self.task_1Hz['task_NDH_1Hz']
        task_dh = self.task_1Hz['task_DH_1Hz']

        count_ndh = np.concatenate(count_ndh, axis=None)
        count_dh = np.concatenate(count_dh, axis=None)
        task_ndh = np.concatenate(task_ndh, axis=None)
        task_dh = np.concatenate(task_dh, axis=None)

        for task in protocol_tasks:
            counts_ndh = extract_all_values_with_label(count_ndh, task_ndh, task)
            counts_nd = extract_all_values_with_label(count_dh, task_dh, task)
            self.count_per_task_ndh[task], self.count_per_task_dh[task] = counts_ndh, counts_nd
    
    def plot_polar_histogram(self, primitive_task='primitive'):
        """
        Plot a half circle polar histogram of the self.pitch_per_primitive_ndh and self.pitch_per_primitive_dh
        with subplots for each primitive.
        """
        labels = list(self.pitch_per_task_dh.keys())
        if primitive_task == 'primitive':
            labels = list(self.pitch_per_primitive_ndh.keys())
            print("Not plotting "+labels.pop()+" as it is not a primitive")
        # Set the number of rows and columns for the subplots
        num_rows = 2
        num_cols = len(labels)
        # Create a figure with subplots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(30, 6))#, subplot_kw={'projection': 'polar'} wanted to make it polar but it didn't work
        # Iterate over the primitives
        for i, label in enumerate(labels):
            pitch_ndh = []
            pitch_dh = []
            if primitive_task == 'task':
                pitch_ndh = self.pitch_per_task_ndh[label]
                pitch_dh = self.pitch_per_task_dh[label]
            elif primitive_task == 'primitive':
                pitch_ndh = self.pitch_per_primitive_ndh[label]
                pitch_dh = self.pitch_per_primitive_dh[label]
            else:
                raise ValueError("Invalid value for 'primitive_task'. Please use 'primitive' or 'task'.")
            # Compute the histogram of pitch values for NDH
            hist_ndh, bins_ndh = np.histogram(pitch_ndh, bins=180, range=(-90, 90), density=True)
            # Compute the histogram of pitch values for DH
            hist_dh, bins_dh = np.histogram(pitch_dh, bins=180, range=(-90, 90), density=True)
            # Compute the bin centers for NDH
            bin_centers_ndh = 0.5 * (bins_ndh[:-1] + bins_ndh[1:])
            # Compute the bin centers for DH
            bin_centers_dh = 0.5 * (bins_dh[:-1] + bins_dh[1:])
            # Plot the polar histogram for NDH
            ax = axes[0, i]
            ax.plot(bin_centers_ndh, hist_ndh, color=thesis_style.get_label_colours()[label])
            ax.set_title(f'{label} NDH')
            ax.set_xticks([-90, -45, 0, 45, 90])
            ax.set_xticklabels(['-90°', '-45°', '0°', '45°', '90°'])
            ax.set_yticks([])
            ax.set_ylim([0, np.max(hist_ndh) * 1.1])
            #ax.spines['polar'].set_visible(False)
            # Plot the polar histogram for DH
            ax = axes[1, i]
            ax.plot(bin_centers_dh, hist_dh, color=thesis_style.get_label_colours()[label])
            ax.set_title(f'{label} DH')
            ax.set_xticks([-90, -45, 0, 45, 90])
            ax.set_xticklabels(['-90°', '-45°', '0°', '45°', '90°'])
            ax.set_yticks([])
            ax.set_ylim([0, np.max(hist_dh) * 1.1])
            #ax.spines['polar'].set_visible(False)
        
        # Adjust the spacing between subplots
        fig.tight_layout(pad=3)
        fig.suptitle('Elevation Histogram per Label', fontsize=24, y=1.0)
        plt.show()

    def plot_count_histograms_per_task(self):
        """
        Plot the count histograms for each task.
        """
        labels = list(self.count_per_task_dh.keys())
        # Set the number of rows and columns for the subplots
        num_rows = 2
        num_cols = len(labels)
        # Create a figure with subplots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(30, 6))
        # Iterate over the tasks
        for i, label in enumerate(labels):
            count_ndh = self.count_per_task_ndh[label]
            count_dh = self.count_per_task_dh[label]
            # Compute the histogram of count values for NDH
            hist_ndh, bins_ndh = np.histogram(count_ndh, bins=180, range=(0, 180), density=True)
            # Compute the histogram of count values for DH
            hist_dh, bins_dh = np.histogram(count_dh, bins=180, range=(0, 180), density=True)
            # Compute the bin centers for NDH
            bin_centers_ndh = 0.5 * (bins_ndh[:-1] + bins_ndh[1:])
            # Compute the bin centers for DH
            bin_centers_dh = 0.5 * (bins_dh[:-1] + bins_dh[1:])
            # Plot the count histogram for NDH
            ax = axes[0, i]
            ax.plot(bin_centers_ndh, hist_ndh, color=thesis_style.get_label_colours()[label])
            ax.set_title(task_to_formated.get_task_conversion()[label] +' NDH')
            ax.set_yticks([])
            ax.set_ylim([0, np.max(hist_ndh) * 1.1])
            # Plot the count histogram for DH
            ax = axes[1, i]
            ax.plot(bin_centers_dh, hist_dh, color=thesis_style.get_label_colours()[label])
            ax.set_title(task_to_formated.get_task_conversion()[label] +' DH')
            ax.set_yticks([])
            ax.set_ylim([0, np.max(hist_dh) * 1.1])

        # Adjust the spacing between subplots
        fig.tight_layout(pad=3)
        fig.suptitle('Count Histogram per Task', fontsize=24, y=1.0)
        plt.show()
    

    def plot_pitch_per_label(self):
        """
        Plot the average pitch per primitive.
        """
        _ = self.get_pitch_per_primitive()
        pitch_per_label = self.get_pitch_per_functional()
        del pitch_per_label["arm_not_visible"]
        plt.bar(pitch_per_label.keys(), pitch_per_label.values(), color=[thesis_style.get_label_colours()[key] for key in pitch_per_label.keys()])
        plt.xticks(rotation=45)
        plt.ylabel('Average Pitch (degrees)')
        plt.title('Average Pitch per Label')
        plt.tight_layout(rect=[0, 0, 1.5, 1])
        plt.show()