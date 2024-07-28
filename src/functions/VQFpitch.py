import numpy as np
import pandas as pd
import vqf  # Ensure the vqf library is installed and properly imported
from scipy.spatial.transform import Rotation as R

class IMUPitchCalculator:
    def __init__(self, acc, gyro, sampling_freq):
        # Constants for conversions
        g_to_m_s2 = 9.80665  # 1 g = 9.80665 m/s^2
        deg_to_rad = np.pi / 180  # 1 degree = π/180 radians
        
        # Extract and convert acceleration and gyroscope data
        self.accel = acc * g_to_m_s2
        self.gyro = gyro * deg_to_rad
        self.Ts = 1/sampling_freq

    def scalar_first_to_scalar_last(self, quaternions):
        """
        Convert quaternions from scalar-first to scalar-last convention.
        """
        return np.roll(quaternions, -1, axis=1)
        
    def calculate_pitch(self):
        # Calculate quaternions using vqf.offlineVQF
        quaternions = vqf.offlineVQF(self.gyro, self.accel, None, self.Ts)
        quaternions_scalar_last = self.scalar_first_to_scalar_last(quaternions['quat6D'])
        # Convert quaternions to Euler angles
        r = R.from_quat(quaternions_scalar_last)
        euler_angles = r.as_euler('yxz', degrees=True) #TODO think about convention
        
        # Extract pitch from the Euler angles
        self.pitch = euler_angles[:, 1]
        return self.pitch
    
    def cut_recording(self, start_index, end_index):
        # Cut the pitch data
        self.pitch = self.pitch[start_index:end_index]
    
    def plot_pitch(self):
        # Create a DataFrame for pitch data
        pitch_data = pd.DataFrame({'loggingTime(txt)': np.arange(len(self.pitch)) * self.Ts, 'motionPitch(deg)': self.pitch})
        
        # Plot pitch data
        pitch_data.plot(x='loggingTime(txt)', y='motionPitch(deg)', title='Pitch data', figsize=(12, 6))