import pandas as pd
import numpy as np
from scipy import signal

from utilities import *

def read_pitch_from_csv(file_path):
    """
    Read pitch data from a CSV file.
    """
    pitch_data = pd.read_csv(file_path)
    pitch_data = pitch_data[['motionPitch(rad)', 'loggingTime(txt)']]
    pitch_data['motionPitch(deg)'] = pitch_data['motionPitch(rad)'].apply(lambda x: x * 180 / np.pi)
    return pitch_data

def plot_pitch_data(pitch_data):
    """
    Plot pitch data.
    """
    pitch_data.plot(x='loggingTime(txt)', y='motionPitch(deg)', title='Pitch data', figsize=(12, 6))

def estimate_pitch(accl: np.array, farm_inx: int, nwin: int) -> np.array:
    """
    Estimates the pitch angle of the forearm from the accelerometer data.
    Adapted from Balasubramanian 2023
    """
    # Moving averaging using the causal filter
    acclf = signal.lfilter(np.ones(nwin) / nwin, 1, accl, axis=0) if nwin > 1 else accl
    # Compute the norm of the acceleration vector
    acclfn = acclf / np.linalg.norm(acclf, axis=1, keepdims=True)
    return -np.rad2deg(np.arccos(acclfn[:, farm_inx])) + 90

def plot_angles_over_time(angles):
    """
    Plots angles over time.

    Args:
        angles (np.ndarray): Array of angles in degrees.
        sampling_freq (int): Sampling frequency in Hz.

    Returns:
        None.
    """
    
    sampling_freq = 50 #Hz 
    
    # Increase figure size
    plt.figure(figsize=(18, 9))
    
    # Calculate time array based on the length and sampling frequency
    time = np.arange(len(angles)) / sampling_freq

    # Plot angles over time
    plt.plot(time, angles)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Angle (degrees)')
    plt.title('Angles over Time')
    plt.grid(True)
    plt.show()

def plot_normalized_distribution(angles):
    # Compute the histogram of angles
    bins = 50  # Number of bins for the histogram
    hist, edges = np.histogram(angles, bins=bins, density=True)

    # Compute the bin centers
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    # Set the figure size
    plt.figure(figsize=(16, 8))

    # Plot the normalized distribution as a bar plot
    plt.bar(bin_centers, hist, width=np.diff(edges), align='center')

    plt.xlabel('Angle (degrees)')
    plt.ylabel('Normalized Frequency')
    plt.title('Normalized Distribution of Angles')
    plt.show()

def plot_superposed_normalized_distribution(*angle_arrays):
    """
    Plots the superposed normalized distribution of several arrays of angles.

    Args:
        *angle_arrays: Variable number of arrays of angles.

    Returns:
        None (displays the plot).
    """
    # Create a color map for differentiating arrays
    colors = plt.cm.get_cmap('tab10').colors
    
    # Set the figure size
    plt.figure(figsize=(16, 8))
    
    # Save the names of the input arrays for all the angle_arrays
    array_names = [get_array_name(arr) for arr in angle_arrays]
    
    # Plot the normalized distribution for each array
    for i, angles in enumerate(angle_arrays):
        # Compute the histogram of angles
        bins = 50  # Number of bins for the histogram
        hist, edges = np.histogram(angles, bins=bins, density=True)
        
        # Compute the bin centers
        bin_centers = 0.5 * (edges[:-1] + edges[1:])
        
        # Get the array name
        label = array_names[i] if array_names[i] is not None else f"Array_{i+1}"
        
        # Plot the normalized distribution as a line plot
        plt.plot(bin_centers, hist, color=colors[i % len(colors)], label=label)
    
    # Set the axis labels and title
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Normalized Frequency')
    plt.title('Superposed Normalized Distribution of Angles')
    
    # Add a legend to explain the colors
    plt.legend(loc='upper right')
    
    # Show the plot
    plt.show()

class PitchPerPrimitive:
    def __init__(self, label_to_int, initial_path = '../data/CreateStudy'):
        
        self.label_to_int = label_to_int.copy()
        self.pitch_per_primitive_ndh = self.label_to_int.copy()
        self.pitch_per_primitive_dh = self.label_to_int.copy()
        self.mean_pitch_per_primitive = self.label_to_int.copy()

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
        for path in s_json_files:
            gt_dict = extract_fields_from_json_files([path], ['primitive_mask_LW_25Hz', 'primitive_mask_RW_25Hz', 'GT_mask_NDH_25Hz', 'GT_mask_DH_25Hz'])
            pitch_dict_50Hz = extract_fields_from_json_files([path], ['pitch_NDH', 'pitch_DH'])
            primitives_LW.append(gt_dict['primitive_mask_LW_25Hz'])
            primitives_RW.append(gt_dict['primitive_mask_RW_25Hz'])
            GT_NDH_25Hz.append(gt_dict['GT_mask_NDH_25Hz'])
            GT_DH_25Hz.append(gt_dict['GT_mask_DH_25Hz'])
            pitch_NDH_25Hz.append(np.average(pitch_dict_50Hz['pitch_NDH'].reshape(-1, 2), axis=1))
            pitch_DH_25Hz.append(np.average(pitch_dict_50Hz['pitch_DH'].reshape(-1, 2), axis=1))
        primitives = {'primitive_mask_LW_25Hz': primitives_LW, 'primitive_mask_RW_25Hz': primitives_RW}
        gt_functional = {'GT_mask_NDH_25Hz': GT_NDH_25Hz, 'GT_mask_DH_25Hz': GT_DH_25Hz}
        pitch_data = {'pitch_NDH_25Hz': pitch_NDH_25Hz, 'pitch_DH_25Hz': pitch_DH_25Hz}
        self.primitives = primitives
        self.gt_functional = gt_functional
        self.pitch_data = pitch_data

    def extract_all_values_with_label(self, array, label_array, label_of_interest):
        extracted_array = array[label_array == label_of_interest]
            
        return extracted_array#, label_of_interest

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
            mean_pitch = np.mean(self.extract_all_values_with_label(combined_pitch, combined_primitives, value))
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
                pitches_ndh = self.extract_all_values_with_label(pitch_ndh, gt_functional_ndh, value)
                pitches_nd = self.extract_all_values_with_label(pitch_dh, gt_functional_dh, value)
                self.pitch_per_primitive_ndh[key], self.pitch_per_primitive_dh[key] = pitches_ndh, pitches_nd
                continue
            pitches_ndh = self.extract_all_values_with_label(pitch_ndh, primitives_ndh, value)
            pitches_nd = self.extract_all_values_with_label(pitch_dh, primitives_dh, value)
            self.pitch_per_primitive_ndh[key], self.pitch_per_primitive_dh[key] = pitches_ndh, pitches_nd
    
    def plot_polar_histogram(self):
        """
        Plot a half circle polar histogram of the self.pitch_per_primitive_ndh and self.pitch_per_primitive_dh
        with subplots for each primitive.
        """
        # Get the list of primitives
        primitives = list(self.pitch_per_primitive_ndh.keys())
        print("Not plotting "+primitives.pop()+" as it is not a primitive")
        # Set the number of rows and columns for the subplots
        num_rows = 2
        num_cols = len(primitives)
        # Create a figure with subplots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(24, 8))#, subplot_kw={'projection': 'polar'} wanted to make it polar but it didn't work
        # Iterate over the primitives
        for i, primitive in enumerate(primitives):
            # Get the pitch data for NDH and DH
            pitch_ndh = self.pitch_per_primitive_ndh[primitive]
            pitch_dh = self.pitch_per_primitive_dh[primitive]
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
            ax.plot(bin_centers_ndh, hist_ndh, color=thesis_style.get_label_colours()[primitive])
            ax.set_title(f'{primitive} (NDH)')
            ax.set_xticks([-90, -45, 0, 45, 90])
            ax.set_xticklabels(['-90°', '-45°', '0°', '45°', '90°'])
            ax.set_yticks([])
            ax.set_ylim([0, np.max(hist_ndh) * 1.1])
            #ax.spines['polar'].set_visible(False)
            # Plot the polar histogram for DH
            ax = axes[1, i]
            ax.plot(bin_centers_dh, hist_dh, color=thesis_style.get_label_colours()[primitive])
            ax.set_title(f'{primitive} (DH)')
            ax.set_xticks([-90, -45, 0, 45, 90])
            ax.set_xticklabels(['-90°', '-45°', '0°', '45°', '90°'])
            ax.set_yticks([])
            ax.set_ylim([0, np.max(hist_dh) * 1.1])
            #ax.spines['polar'].set_visible(False)
        
        # Adjust the spacing between subplots
        fig.tight_layout(pad=3)
        fig.suptitle('Pitch Histogram per Label', fontsize=24)# Add more space below the title
        fig.subplots_adjust(top=0.9)
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