import math
import os
import numpy as np
from scipy import signal, interpolate, stats
import pandas as pd
import resampy
import csv
from datetime import datetime
from matplotlib.colors import ListedColormap
from ahrs.filters import Mahony, Madgwick
from ahrs.common import orientation
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter, freqz
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import resample
from scipy.spatial.transform import Rotation
from scipy.interpolate import CubicSpline
from agcounts.extract import get_counts
import matplotlib.pyplot as plt

##predefined filter coefficients, as found by Jan Brond
A_coeff = np.array(
    [1, -4.1637, 7.5712,-7.9805, 5.385, -2.4636, 0.89238, 0.06361, -1.3481, 2.4734, -2.9257, 2.9298, -2.7816, 2.4777,
     -1.6847, 0.46483, 0.46565, -0.67312, 0.4162, -0.13832, 0.019852])
B_coeff = np.array(
    [0.049109, -0.12284, 0.14356, -0.11269, 0.053804, -0.02023, 0.0063778, 0.018513, -0.038154, 0.048727, -0.052577,
     0.047847, -0.046015, 0.036283, -0.012977, -0.0046262, 0.012835, -0.0093762, 0.0034485, -0.00080972, -0.00019623])

def pptrunc(data, max_value):
    '''
    Saturate a vector such that no element's absolute value exceeds max_abs_value.
    Current name: absolute_saturate().
      :param data: a vector of any dimension containing numerical data
      :param max_value: a float value of the absolute value to not exceed
      :return: the saturated vector
    '''
    outd = np.where(data > max_value, max_value, data)
    return np.where(outd < -max_value, -max_value, outd)

def trunc(data, min_value):
  
    '''
    Truncate a vector such that any value lower than min_value is set to 0.
    Current name zero_truncate().
    :param data: a vector of any dimension containing numerical data
    :param min_value: a float value the elements of data should not fall below
    :return: the truncated vector
    '''

    return np.where(data < min_value, 0, data)

def runsum(data, length, threshold):
    '''
    Compute the running sum of values in a vector exceeding some threshold within a range of indices.
    Divides the data into len(data)/length chunks and sums the values in excess of the threshold for each chunk.
    Current name run_sum().
    :param data: a 1D numerical vector to calculate the sum of
    :param len: the length of each chunk to compute a sum along, as a positive integer
    :param threshold: a numerical value used to find values exceeding some threshold
    :return: a vector of length len(data)/length containing the excess value sum for each chunk of data
    '''
    
    N = len(data)
    cnt = int(math.ceil(N/length))

    rs = np.zeros(cnt)

    for n in range(cnt):
        for p in range(length*n, length*(n+1)):
            if p<N and data[p]>=threshold:
                rs[n] = rs[n] + data[p] - threshold

    return rs

def counts(data, filesf, B=B_coeff, A=A_coeff):
    '''
    Get activity counts for a set of accelerometer observations.
    First resamples the data frequency to 30Hz, then applies a Butterworth filter to the signal, then filters by the
    coefficient matrices, saturates and truncates the result, and applies a running sum to get the final counts.
    Current name get_actigraph_counts()
    :param data: the vertical axis of accelerometer readings, as a vector
    :param filesf: the number of observations per second in the file
    :param a: coefficient matrix for filtering the signal, as found by Jan Brond
    :param b: coefficient matrix for filtering the signal, as found by Jan Brond
    :return: a vector containing the final counts
    '''
    
    deadband = 0.068
    sf = 30
    peakThreshold = 2.13
    adcResolution = 0.0164
    integN = 10
    gain = 0.965

    #Resample at 30Hz
    if filesf>sf:
        data = resampy.resample(np.asarray(data), filesf, sf)

    # Step1 : Aliasing filtering
    B2, A2 = signal.butter(4, np.array([0.01, 7])/(sf/2), btype='bandpass')
    dataf = signal.filtfilt(B2, A2, data)
    B = B * gain

    # Step2 : Actigraph filtering 
    fx8up = signal.lfilter(B, A, dataf)

    # Step3 and 4: Down sampling 10Hz (slicing 1/3) + truncation using peakThreshold (2.13g)
    fx8 = pptrunc(fx8up[::3], peakThreshold) #downsampling is replaced by slicing with step parameter
    
    # Step 5 to 8
    return runsum(np.floor(trunc(np.abs(fx8), deadband)/adcResolution), integN, 0), fx8


def get_counts_brond(data):
    filesf = 50
    data_accel = data[['acc_x', 'acc_y', 'acc_z']]

    # calculate counts per axis
    c1_1s, processed_data_x = counts(data_accel.iloc[:, 0], filesf)
    c2_1s, processed_data_y = counts(data_accel.iloc[:, 1], filesf)
    c3_1s, processed_data_z = counts(data_accel.iloc[:, 2], filesf)

    # combine counts in pandas DataFrame
    axis_counts = pd.DataFrame(data={'Axis 1': c1_1s, 'Axis 2': c2_1s, 'Axis 3': c3_1s})
    axis_counts = axis_counts.astype(int)

    # combine processed data
    processed_data = pd.DataFrame(data={
        'prepro_acc_x': processed_data_x,
        'prepro_acc_y': processed_data_y,
        'prepro_acc_z': processed_data_z
    })
    processed_data = processed_data.astype(float)
    
    # Calculate magnitude and add it as a new column
    axis_counts['AC Brond'] = (axis_counts['Axis 1'] ** 2 + axis_counts['Axis 2'] ** 2 + axis_counts['Axis 3'] ** 2) ** 0.5
    
    return axis_counts, processed_data


def plot_actigraph_count(data):
    time = range(data.shape[0])  # Assuming data has shape (number_count, 4)
    time_in_seconds = [t for t in time]  # Epochs correspond to seconds

    plt.figure(figsize=(10, 6))
    for i in range(data.shape[1] - 1):  # Exclude the last column (AC)
        axis_name = data.columns[i]  # Extract axis name from DataFrame
        plt.plot(time_in_seconds, data.iloc[:, i], label=axis_name)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Activity Count')
    plt.title('Actigraph Activity Count per Axis')
    plt.legend()
    plt.grid(True)
    plt.show()

    last_column_name = data.columns[-1]  # Extract the name of the last column
    plt.figure(figsize=(10, 6))
    plt.plot(time_in_seconds, data.iloc[:, -1])  # Plot the last column
    plt.xlabel('Time (s)')
    plt.ylabel(last_column_name)
    plt.title('Actigraph Magnitude ({})'.format(last_column_name))
    plt.grid(True)
    plt.show()


def get_counts_neishabouri(df, freq: int = 50, epoch: int = 1, fast: bool = False, verbose: bool = False, time_column: str = None):
    # verbose: Print diagnostic messages
    if verbose:
        print("Converting to array", flush=True)
    raw = df[["acc_x", "acc_y", "acc_z"]]
    raw = np.array(raw)
    
    if verbose:
        print("Getting Counts", flush=True)
    counts = get_counts(raw, freq=freq, epoch=epoch, fast=fast, verbose=verbose)
    del raw
    counts = pd.DataFrame(counts, columns=["Axis 1", "Axis 2", "Axis 3"])
    counts["AC Neishabouri"] = (counts["Axis 1"] ** 2 + counts["Axis 2"] ** 2 + counts["Axis 3"] ** 2) ** 0.5
    
    ts = pd.DataFrame()  # Initialize ts as an empty DataFrame
    if time_column is not None:
        ts = df[time_column]
        ts = pd.to_datetime(ts)
        time_freq = str(epoch) + "S"
        ts = ts.dt.round(time_freq)
        ts = ts.unique()
        ts = pd.DataFrame(ts, columns=[time_column])
        ts = ts[0:counts.shape[0]]
        counts = pd.concat([ts, counts], axis=1)
    
    return counts
