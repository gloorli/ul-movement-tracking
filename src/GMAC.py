#https://github.com/biorehab/upper-limb-use-assessment/tree/main
import os
import numpy as np
import pandas as pd
import scipy.signal as signal
from ahrs.filters import Madgwick
from ahrs.common import orientation
from gm_function import compute_euler_angles
from imu_video_synch import get_datetime_timestamp_Axivity, create_timestamps

def get_continuous_segments(df):
    # returns a list of continuous sections (as dataframes) from the original dataframe

    time_diff = np.array([pd.Timedelta(diff).total_seconds() for diff in np.diff(df.index.values)])
    inx = np.sort(np.append(np.where(time_diff > 60)[0], -1))
    dfs = [df.iloc[inx[i] + 1:inx[i + 1] + 1] if i + 1 < len(inx) else df.iloc[inx[-1] + 1:] for i in
           np.arange(len(inx))]
    return dfs

def resample(df, current_fs, new_fs):
    dfs = get_continuous_segments(df)
    dfs = [df.resample(str(round(1 / new_fs, 2)) + 'S', label='right', closed='right').mean() for df in dfs]
    df = pd.concat(dfs)
    df.index.name = 'time'
    return df

def bandpass(x, fs=50, order=4):
    sos = signal.butter(order, [0.25, 2.5], 'bandpass', fs=fs, output='sos', analog=False)
    filtered = signal.sosfilt(sos, x)
    return filtered

def compute_vector_magnitude(df):
    df = resample(df, 50, 30)
    op_df = pd.DataFrame(index=df.index)

    gyr = np.array(df[['gx', 'gy', 'gz']])
    acc = np.array(df[['ax', 'ay', 'az']])

    g = np.array([0, 0, 1])
    ae = np.empty([len(acc), 3])

    mg = Madgwick(frequency=30, beta=0.5)
    q = np.tile([1., 0., 0., 0.], (len(acc), 1))

    r = orientation.q2R(mg.updateIMU(q[0], gyr[0], acc[0]))
    ae[0] = np.matmul(r, acc[0]) - g

    for i in range(1, len(acc)):
        q[i] = mg.updateIMU(q[i - 1], gyr[i], acc[i])
        r = orientation.q2R(q[i])
        ae[i] = np.matmul(r, acc[i]) - g

    op_df['ax'] = bandpass(np.nan_to_num(ae[:, 0]), fs=30)
    op_df['ay'] = bandpass(np.nan_to_num(ae[:, 1]), fs=30)
    op_df['az'] = bandpass(np.nan_to_num(ae[:, 2]), fs=30)
    op_df = resample(op_df, 30, 10)

    op_df['ax'] = np.where(np.absolute(op_df['ax'].values) < 0.068, 0, op_df['ax'].values) / 0.01664
    op_df['ay'] = np.where(np.absolute(op_df['ay'].values) < 0.068, 0, op_df['ay'].values) / 0.01664
    op_df['az'] = np.where(np.absolute(op_df['az'].values) < 0.068, 0, op_df['az'].values) / 0.01664

    dfs = get_continuous_segments(op_df)
    dfs = [df.resample(str(1) + 'S').sum() for df in dfs]
    op_df = pd.concat(dfs)
    op_df.index.name = 'time'
    op_df = op_df.fillna(0)

    op_df['a_mag'] = [np.linalg.norm(x) for x in np.array(op_df[['ax', 'ay', 'az']])]
    op_df['counts'] = [np.round(x) for x in op_df['a_mag'].rolling(5).mean()]
    return op_df[['counts']]

def add_datetime_index(df, fs=50):
    IMU_start_timestamp, IMU_end_timestamp = get_datetime_timestamp_Axivity(df.index[0], df.index[-1])
    # Get all timestamps
    timestamps_array = create_timestamps(IMU_start_timestamp, IMU_end_timestamp, fs)
    # Add timestamps to raw data (needed for trimming step)
    df.reset_index(drop=True, inplace=True)
    return pd.concat([timestamps_array, df], axis=1)

def GMAC(IMU_data, count_threshold=0, functional_range=30):
    pitch = resample(IMU_data[['pitch']], 50, 1)
    counts = compute_vector_magnitude(IMU_data)
    gmac = pd.merge(pitch, counts, on='time')
    gmac['pred'] = [1 if np.abs(pitch) < functional_range and count > count_threshold else 0 for pitch, count in zip(gmac['pitch'], gmac['counts'])]
    return gmac.reset_index()

def main():
    filepath = 'data/CreateStudy/S001' #TODO: change the path to the file
    filename = 'S001_LW.csv' #TODO: change the filename
    df_for_Subash = pd.read_csv(os.path.join(filepath, filename))
    df_for_Subash.columns = ['time', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']
    df_for_Subash.set_index('time', inplace=True)
    _, pitch_mad, _ = compute_euler_angles(df_for_Subash[['ax', 'ay', 'az']], df_for_Subash[['gx', 'gy', 'gz']], fs=50)
    df_for_Subash['pitch'] = pitch_mad

    df_for_Subash = add_datetime_index(df_for_Subash, fs=50)
    df_for_Subash.set_index('timestamp', inplace=True)

    GMACdf = GMAC(df_for_Subash)
    GMACdf.to_csv(os.path.join(filepath, 'GMACdf.csv'))
    return GMACdf

if __name__ == "__main__":
    main()