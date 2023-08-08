import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import pandas as pd
from ahrs.filters import Madgwick, Mahony
from math import pi, nan
import csv
from datetime import datetime
from matplotlib.colors import ListedColormap
from ahrs.common import orientation
import os
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal, interpolate, stats
import matplotlib.pyplot as plt
from scipy.signal import resample, savgol_filter, freqz, lfilter
from scipy.spatial.transform import Rotation
from scipy.stats import circmean
from scipy.interpolate import CubicSpline
import inspect
from utilities import *
from ahrs.filters import Mahony

#Usefull constant
GRAVITY_CONSTANT = 9.81


def detrend_angle(angle):
    """
    Detrends the input angle by removing the linear trend.

    Args:
        angle (ndarray): Array of angle values.

    Returns:
        ndarray: Detrended angle values.
    """
    time = np.arange(len(angle))
    slope, intercept = np.polyfit(time, angle, 1)
    detrended_angle = angle - (slope * time + intercept)
    return detrended_angle


def quaternion_to_euler(quaternions):
    # Ensure to use Numpy array
    quaternions = np.array(quaternions)

    # Extract quaternion components
    w = quaternions[:, 0]
    x = quaternions[:, 1]
    y = quaternions[:, 2]
    z = quaternions[:, 3]

    # Roll (Bank)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))

    # Pitch (Elevation)
    sin_pitch = 2 * (w * y - z * x)
    # Check for gimbal lock avoidance
    mask = np.abs(sin_pitch) > 1
    pitch = np.where(mask, np.sign(sin_pitch) * np.pi / 2, np.arcsin(sin_pitch))

    # Yaw (Heading)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

    # Convert angles to degrees
    roll = np.degrees(roll)
    pitch = np.degrees(pitch)
    yaw = np.degrees(yaw)

    return roll, pitch, yaw


def imu2quat(acc, gyro, *mag):
    """
    Convert accelerometer, gyroscope, and magnetometer data into quaternions using the Madgwick filter.

    Args:
        acc: Accelerometer data (list or array).
        gyro: Gyroscope data (list or array).
        *mag: Magnetometer data (optional, list or array).

    Returns:
        quat: Quaternion representing the estimated orientation.

    """
    # Convert into Numpy array
    acc = np.array(acc)
    gyro = np.array(gyro)
    
    # Convert gyro to rad/s
    gyro = gyro * (pi / 180)
    
    # Convert acc to m/s^-2
    acc = acc * GRAVITY_CONSTANT
    
    if mag:
        # MARG case: acc, gyro, and mag data are provided
        mag_data = np.array(*mag)
        
        # Convert mag from micro to milli tesla
        mag_data = mag_data / 1000
        
        madgwick = Madgwick(gyr=gyro, acc=acc, mag=mag_data, frequency=50)
        quat = madgwick.Q
    else:
        # Non-MARG case: only acc and gyro data are provided
        madgwick = Madgwick(gyr=gyro, acc=acc, frequency=50)
        quat = madgwick.Q
    
    return quat


def compute_euler_angles(acc, gyro, fs, quat=None, mag=None, mahony=False):
    """
    Compute Euler angles from accelerometer and gyroscope data.

    Args:
        acc (np.ndarray): Accelerometer data.
        gyro (np.ndarray): Gyroscope data.
        fs (int): Sampling frequency.
        quat (np.ndarray, optional): Quaternion data from sensor. Defaults to None.
        mag (np.ndarray, optional): Magnetometer data. Defaults to None.
        mahony (bool, optional): Flag to indicate Mahony MARG fusion. Defaults to False.

    Returns:
        tuple: Euler angles (roll, pitch, yaw).
    """

    # Ensure to use Numpy arrays
    acc = np.array(acc)
    gyro = np.array(gyro)

    imu_madgwick = False
    sensor = False

    if quat is not None:
        print('Using quaternion from sensor')
        q = quat.to_numpy()
        sensor = True
    else:
        if mag is not None:
            mag = np.array(mag)
            if mahony:
                print('Using Mahony MARG fusion')
                mahony = Mahony(gyr=gyro, acc=acc, mag=mag, frequency=fs)
                q = mahony.Q
            else:
                # Orientation estimation using MARG
                print('Using Madgwick MARG fusion')
                q = imu2quat(acc, gyro, mag)
        else:
            if mahony:
                print('Using IMU Mahony with acc and gyro only')
                mahony = Mahony(gyr=gyro, acc=acc, frequency=fs)
                q = mahony.Q
            else:
                # Orientation estimation using IMU Madgwick
                print('Using IMU Madgwick with acc and gyro only')
                imu_madgwick = True
                q = imu2quat(acc, gyro)

    # Calculate Euler angles
    roll, pitch, yaw = quaternion_to_euler(q)

    # Remove linear trend if Madgwick IMU acc + gyro or sensor
    if imu_madgwick or sensor:
        yaw = detrend_angle(yaw)
    
    # Inverse sign 
    roll = -1 * roll
    pitch = -1 * pitch
    yaw = -1 * yaw

    return roll, pitch, yaw


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

    
def segment_data_gm(data, window_size=2, overlap=0.75, sampling_rate=50):
    
    # Calculate the number of samples in a window
    window_samples = int(window_size * sampling_rate)
    
    # Calculate the number of overlapping samples
    overlap_samples = int(window_samples * overlap)
    
    # Calculate the number of windows
    num_windows = int(np.ceil(len(data) / (window_samples - overlap_samples)))
    
    # Pad the data with NaN values if necessary
    padding_size = (window_samples - len(data) % window_samples) % window_samples
    padded_data = np.concatenate([data, np.full(padding_size, np.nan)])
    
    # Create a list to hold the segmented data arrays
    segmented_data_list = []
    
    # Fill in the segmented data array with the sliding window approach
    for i in range(num_windows):
        start_index = i * (window_samples - overlap_samples)
        end_index = start_index + window_samples
        segmented_data_list.append(padded_data[start_index:end_index])
    
    # Convert the list of arrays to a numpy array of arrays
    segmented_data = np.array(segmented_data_list, dtype=object)
    
    return segmented_data


def gm_algorithm (pitch, yaw, functional_space):
    """
    Compute Gross Movement (GM) based on yaw and pitch angles.

    Args:
    pitch (array-like): Array of pitch angles.
    yaw (array-like): Array of yaw angles.

    Returns:
    gm_epochs (ndarray): Array indicating GM epochs (1 for GM, 0 otherwise).
    """

    epoch_size_leuenberger = 2  # s
    overlap_leuenberger = 0.75  # 75%

    pitch = np.array(pitch)
    yaw = np.array(yaw)

    # Epoch of 2s with 75% overlapping
    segmented_theta = segment_data_gm(pitch)
    segmented_yaw = segment_data_gm(yaw)

    # Compute the number of epochs
    num_epochs = segmented_theta.shape[0]

    # Initialize an empty array to hold the GM for each epoch
    gm_epochs = np.zeros(num_epochs)

    for i in range(num_epochs):

        # Extract the theta and yaw epochs
        epoch_thetas = segmented_theta[i]
        epoch_psis = segmented_yaw[i]

        # Compute the overall absolute change in yaw and pitch angles inside the epoch
        delta_psi = np.abs(np.nanmax(epoch_psis) - np.nanmin(epoch_psis))
        delta_theta = np.abs(np.nanmax(epoch_thetas) - np.nanmin(epoch_thetas))

        # Compute the absolute pitch of the forearm
        epoch_pitch_mean  = np.mean(epoch_thetas)

        # Check if the criteria for GM are met
        # only the pitch angle is concerned by this functional space
        if ((delta_psi + delta_theta) > 30) and (np.abs(epoch_pitch_mean ) < functional_space):
            gm_epochs[i] = 1

    return gm_epochs


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
    

def plot_gm_scores(gm_scores, frequency=2):
    time = np.arange(len(gm_scores)) / frequency
    
    # Set the figure size
    plt.figure(figsize=(16, 8))
    
    plt.bar(time, gm_scores, width=1/frequency, align='edge', color='blue', edgecolor='green')
    plt.xlabel('Time (seconds)')
    plt.ylabel('GM Scores')
    plt.title('GM Scores Over Recording Time')
    plt.yticks([0, 1])
    plt.ylim([-0.1, 1.1])
    plt.show()


def compute_GMAC(pitch_mad_50Hz, AC_1Hz, ac_threshold = 0, functional_space = 30):

    # Compute windows of 1 seconds of data for the pitch angles
    theta_per_epoch = window_data(pitch_mad_50Hz, original_sampling_frequency = 50, window_length_seconds = 1)
    
    num_epoch_pitch = len(theta_per_epoch)
    
    # Initialize an array to store the mean pitch per epoch
    theta_mean_per_epoch = np.zeros(num_epoch_pitch)
    
    # Iterate over each epoch and compute the mean pitch angle
    for i in range(num_epoch_pitch):
        theta_mean_per_epoch[i] = np.mean(theta_per_epoch[i])
    
    # Ensure AC and mean pitch per epoch have same size 
    theta_mean_per_epoch, AC_1Hz = remove_extra_elements(theta_mean_per_epoch, AC_1Hz)
    
    # Number of epochs GMAC 
    num_epoch_gmac = len(theta_mean_per_epoch)
    
    # Initialize an array to store GMAC score @ 1Hz
    GMAC = np.zeros(num_epoch_gmac)
    
    for i in range(num_epoch_gmac):
        if ((np.abs(theta_mean_per_epoch[i]) < functional_space) and (AC_1Hz[i] > ac_threshold)):
            GMAC[i] = 1
    return GMAC 