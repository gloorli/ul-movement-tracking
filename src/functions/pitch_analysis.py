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
    def __init__(self, initial_path = '../data/CreateStudy'):
        
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
        for path in s_json_files:
            primitive_dict = extract_fields_from_json_files([path], ['primitive_mask_LW_25Hz', 'primitive_mask_RW_25Hz'])
            pitch_dict_50Hz = extract_fields_from_json_files([path], ['pitch_NDH', 'pitch_DH'])
            primitives_LW.append(primitive_dict['primitive_mask_LW_25Hz'])
            primitives_RW.append(primitive_dict['primitive_mask_RW_25Hz'])
            pitch_NDH_25Hz.append(np.average(pitch_dict_50Hz['pitch_NDH'].reshape(-1, 2), axis=1))
            pitch_DH_25Hz.append(np.average(pitch_dict_50Hz['pitch_DH'].reshape(-1, 2), axis=1))
        primitives = {'primitive_mask_LW_25Hz': primitives_LW, 'primitive_mask_RW_25Hz': primitives_RW}
        pitch_data = {'pitch_NDH_25Hz': pitch_NDH_25Hz, 'pitch_DH_25Hz': pitch_DH_25Hz}
        self.primitives = primitives
        self.pitch_data = pitch_data


    def get_pitch_per_primitive(self):
        """
        Get the pitch data per primitive.
        """
        pitch_per_primitive = {}
        for primitive in self.primitives:
            start_time = primitive['start_time']
            end_time = primitive['end_time']
            pitch_per_primitive[primitive['name']] = self.pitch_data[(self.pitch_data['loggingTime(txt)'] >= start_time) & (self.pitch_data['loggingTime(txt)'] <= end_time)]
        return pitch_per_primitive

    def plot_pitch_per_primitive(self):
        """
        Plot pitch data per primitive.
        """
        pitch_per_primitive = self.get_pitch_per_primitive()
        for primitive_name, pitch_data in pitch_per_primitive.items():
            pitch_data.plot(x='loggingTime(txt)', y='motionPitch(deg)', title=f'Pitch data for {primitive_name}', figsize=(12, 6))