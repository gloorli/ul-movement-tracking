#Adapted from https://github.com/biorehab/upper-limb-use-assessment/tree/main
import pandas as pd
from ahrs.filters import Madgwick
from ahrs.common import orientation
import scipy.signal as signal
import numpy as np
import sys


def get_continuous_segments(df):
    # returns a list of continuous sections (as dataframes) from the original dataframe

    time_diff = np.array([pd.Timedelta(diff).total_seconds() for diff in np.diff(df.index.values)])
    inx = np.sort(np.append(np.where(time_diff > 60)[0], -1))
    dfs = [df.iloc[inx[i] + 1:inx[i + 1] + 1] if i + 1 < len(inx) else df.iloc[inx[-1] + 1:] for i in
           np.arange(len(inx))]
    return dfs


def confmatrix(pred, target):
    n = len(pred)
    notpred = np.logical_not(pred)
    nottarget = np.logical_not(target)
    tp = (np.logical_and(pred, target).sum()) / n
    fp = (np.logical_and(pred, nottarget).sum()) / n
    fn = (np.logical_and(notpred, target).sum()) / n
    tn = (np.logical_and(notpred, nottarget).sum()) / n
    acc = (tp + tn)
    pfa = tp + fn
    pother = tp + fp
    x = ((pfa + pother) / 2)
    pe = 2 * x * (1 - x)
    gwet = (acc - pe) / (1 - pe)
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    prec = tp / (tp + fp) if (tp + fp) else np.nan
    f1 = (2 * sens * prec) / (sens + prec) if (sens + prec) else np.nan
    bal = (sens + spec) / 2
    results = {'true positive': tp,
               'false positive': fp,
               'false negative': fn,
               'true negative': tn,
               'accuracy': acc,
               'gwets ac1 score': gwet,
               'sensitivity': sens,
               'specificity': spec,
               'precision': prec,
               'f1 score': f1,
               'balanced accuracy': bal}
    results = pd.DataFrame.from_dict({k: [v] for k, v in results.items()})
    return results


def bandpass(x, fs=50, order=4):
    sos = signal.butter(order, [0.25, 2.5], 'bandpass', fs=fs, output='sos', analog=False)
    filtered = signal.sosfilt(sos, x)
    return filtered


def resample(df, current_fs, new_fs):
    dfs = get_continuous_segments(df)
    dfs = [df.resample(str(round(1 / new_fs, 2)) + 'S', label='right', closed='right').mean() for df in dfs]
    df = pd.concat(dfs)
    df.index.name = 'time'
    return df


def compute_vector_magnitude(df, gravitation_compensation='Subash'):
    df = resample(df, 50, 30)
    op_df = pd.DataFrame(index=df.index)

    gyr = np.array(df[['gx', 'gy', 'gz']])
    acc = np.array(df[['ax', 'ay', 'az']])

    ae = np.empty([len(acc), 3])
    if gravitation_compensation == 'Subash':
        g = np.array([0, 0, 1])

        mg = Madgwick(frequency=30, beta=0.5)
        q = np.tile([1., 0., 0., 0.], (len(acc), 1))

        r = orientation.q2R(mg.updateIMU(q[0], gyr[0], acc[0]))
        ae[0] = np.matmul(r, acc[0]) - g

        for i in range(1, len(acc)):
            q[i] = mg.updateIMU(q[i - 1], gyr[i], acc[i])
            r = orientation.q2R(q[i])
            ae[i] = np.matmul(r, acc[i]) - g
    elif gravitation_compensation == 'VQF':
        from functions.VQFpitch import vqf_gravitation_compensation
        ae = vqf_gravitation_compensation(acc, gyr)
    else:
        raise ValueError(f"Invalid parameter: {gravitation_compensation}. Use 'Subash' or 'VQF' instead.")

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


def get_GMAC(data):
    pitch = resample(data[['pitch']], 50, 1)
    counts = compute_vector_magnitude(data)
    gmac = pd.merge(pitch, counts, on='time')
    gmac['pred'] = [1 if np.abs(pitch) < 30 and count > 0 else 0 for pitch, count in zip(gmac['pitch'], gmac['counts'])]
    return gmac.reset_index()


def read_data(subject_type):
    """
    Reads raw data from 'subject_type' folder
    """
    if subject_type == 'patient':
        left = pd.read_csv(subject_type + '/data/affected.csv', parse_dates=['time'], index_col='time')
        right = pd.read_csv(subject_type + '/data/unaffected.csv', parse_dates=['time'], index_col='time')
    elif subject_type == 'control':
        left = pd.read_csv(subject_type + '/data/left.csv', parse_dates=['time'], index_col='time')
        right = pd.read_csv(subject_type + '/data/right.csv', parse_dates=['time'], index_col='time')
    else:
        raise Exception(f"Invalid parameter: {subject_type}. Use 'control' or 'patient' instead.")
    return left, right


def generate_hybrid_gmac_output(subject_type):
    """
    Generates and saves classifier output using modified GM score for 'subject_type' dataset
    :param subject_type: 'control', 'patient' - reads data and features from folder
    """
    left, right = read_data(subject_type)
    # modified GM score
    sys.stdout.write('Generating modified GM scores...')
    gml = get_GMAC(left).rename(columns={'pred': 'l'})
    gmr = get_GMAC(right).rename(columns={'pred': 'r'})

    gm = pd.merge(gml, gmr, on='time').set_index('time')
    gm.to_csv(subject_type + '/classifier outputs/gmac.csv')
    sys.stdout.write('Done \n')