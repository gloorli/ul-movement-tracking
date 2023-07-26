import numpy as np
import pandas as pd
from extract_mask_from_video import *
import cv2
from datetime import *
from datetime import datetime, timedelta
import math


def get_participant_paths(participant_id):
    # Specify the folder name
    study_folder = '../CreateStudy'  # Go one step back to access the CreateStudy folder

    # Create the path to the corresponding data
    path = os.path.join(study_folder, participant_id)

    # Specify the file names
    video_filename = participant_id + '.MOV'
    imu_filename = participant_id + '.mat'

    # Construct the full file paths
    video_path = os.path.join(path, video_filename)
    imu_path = os.path.join(path, imu_filename)

    return path, video_path, imu_path


def extract_metadata(video):
    # Use ffprobe command to extract necessary metadata from the media file
    probe = ffmpeg.probe(video)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    tags = video_stream['tags']
    
    # Number of frames
    nb_frames = int(video_stream['nb_frames'])
    
    # Frames per second
    avg_frame_rate = video_stream['avg_frame_rate']
    avg_frame_rate = avg_frame_rate.split('/')
    avg_frame_rate = int(avg_frame_rate[0])
    
    # Duration of the recording in seconds
    duration = float(video_stream['duration'])

    if 'creation_time' in tags:
        # Starting date of recording format '%Y-%m-%dT%H:%M:%S.%fZ'
        creation_time = tags['creation_time']
        # Get an array of timestamps format '%Y-%m-%dT%H:%M:%S.%fZ' over time of the recording
        timestamps = []
        for frame_number in range(nb_frames):
            timestamp = datetime.strptime(creation_time, '%Y-%m-%dT%H:%M:%S.%fZ')
            timestamp += timedelta(seconds=(frame_number / avg_frame_rate))
            formatted_timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')  # Include milliseconds in the format
            timestamps.append(formatted_timestamp)
    else:
        # Handle the case when 'creation_time' key is not present
        creation_time = None
        timestamps = None
        
    return nb_frames, avg_frame_rate, duration, creation_time, np.array(timestamps)


def trim_video(video, start_time, end_time, output_folder):
    try:
        # Generate the output file name
        output_file = os.path.join(output_folder, f"trimmed_{os.path.basename(video)}")

        # Check if the output file already exists
        if os.path.exists(output_file):
            print("Trimmed video already exists. Skipping trimming.")
            return output_file

        # Construct the ffmpeg command
        command = f"ffmpeg -ss {start_time} -to {end_time} -i {video} -c copy {output_file}"

        # Execute the ffmpeg command
        subprocess.run(command, shell=True, check=True)

        return output_file

    except subprocess.CalledProcessError as e:
        print(f"Error trimming video: {str(e)}")


def convert_from_mov_to_mp4(video, output_folder):
    try:
        # Generate the output file name
        output_file = os.path.join(output_folder, f"converted_{os.path.basename(video)}.mp4")

        # Check if the output file already exists
        if os.path.exists(output_file):
            print("Converted video already exists. Skipping conversion.")
            return output_file

        # Construct the ffmpeg command with specified video encoding and format
        command = f"ffmpeg -i {video} -c:v libx264 -crf 23 -preset medium -c:a aac -b:a 128k {output_file}"

        # Execute the ffmpeg command
        subprocess.run(command, shell=True, check=True)

        # Check the output file size
        output_file_size = os.path.getsize(output_file)
        max_file_size = 256 * 1024 * 1024  # 256 MB
        if output_file_size > max_file_size:
            print("Warning: The converted video size exceeds 256 MB.")

        return output_file

    except subprocess.CalledProcessError as e:
        print(f"Error converting video: {str(e)}")

        
def on_trackbar(position):
    # Set the frame number based on the trackbar position
    global frame_number
    frame_number = position


def read_video_frames(video_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Get the total number of frames in the video
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create a window to display the video
    cv2.namedWindow("Frame")

    # Create a trackbar to navigate the frames
    trackbar_name = "Frame Number"
    cv2.createTrackbar(trackbar_name, "Frame", 0, total_frames - 1, on_trackbar)

    # Set the initial frame number
    frame_number = 0
    cv2.setTrackbarPos(trackbar_name, "Frame", frame_number)

    while True:
        # Set the video frame position to the desired frame number
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # Read the frame
        success, frame = video.read()

        if success:
            # Display the frame number over the total frame number
            frame_text = f"Frame: {frame_number}/{total_frames}"
            cv2.putText(frame, frame_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the frame
            cv2.imshow("Frame", frame)

        # Wait for a key press
        key = cv2.waitKey(1)

        # Exit the loop if 'q' is pressed
        if key == ord('q'):
            break

        # Update the frame number based on the trackbar position
        frame_number = cv2.getTrackbarPos(trackbar_name, "Frame")

        # If any other key is pressed, go to the next frame
        if key != -1:
            frame_number += 1

            # Clamp the frame number to stay within the valid range
            frame_number = max(0, min(frame_number, total_frames - 1))

            # Update the trackbar position
            cv2.setTrackbarPos(trackbar_name, "Frame", frame_number)

    # Release the video file
    video.release()
    cv2.destroyAllWindows()


def get_trimmed_video_timestamps(frame_start, frame_end, timestamps): 
    video_start_trim = timestamps[frame_start]
    video_end_trim = timestamps[frame_end]
    
    # Convert to datetime format 
    video_start_timestamp = datetime.strptime(video_start_trim, '%Y-%m-%d %H:%M:%S.%f')
    video_end_timestamp = datetime.strptime(video_end_trim, '%Y-%m-%d %H:%M:%S.%f')
    trimmed_video_duration = video_end_timestamp-video_start_timestamp
    
    return video_start_timestamp, video_end_timestamp,trimmed_video_duration


def convert_seconds_to_time(total_seconds):
    # Create a timedelta object with the total seconds
    time_delta = timedelta(seconds=total_seconds)

    # Extract the hours, minutes, seconds, and milliseconds from the timedelta object
    hours = int(time_delta.total_seconds()) // 3600
    minutes = (int(time_delta.total_seconds()) % 3600) // 60
    seconds = int(time_delta.total_seconds()) % 60
    milliseconds = time_delta.microseconds // 1000

    # Format the time string with the desired format
    formatted_time = "{:02d}:{:02d}:{:02d}.{:03d}".format(hours, minutes, seconds, milliseconds)
    return formatted_time


def get_trimming_times(frame_start, frame_end): 
    fps = 25
    time_start_cutting = frame_start/fps
    time_end_cutting = frame_end/fps
    start_trimming_time = convert_seconds_to_time(time_start_cutting)
    end_trimming_time = convert_seconds_to_time(time_end_cutting)
    return start_trimming_time, end_trimming_time


def get_datetime_timestamp(header): 
    
    format_str = '%Y-%m-%d %H:%M:%S'
    
    start_timestamp = pd.to_datetime(header['startStr'])
    end_timestamp = pd.to_datetime(header['stopStr'])
    start_datetime_str = start_timestamp.dt.strftime('%Y-%m-%d %H:%M:%S.%f').iloc[0]
    end_datetime_str = end_timestamp.dt.strftime('%Y-%m-%d %H:%M:%S.%f').iloc[0]
    
    print(start_datetime_str, end_datetime_str)
    # Extract milliseconds from the string
    start_milliseconds = int(start_datetime_str.split('.')[-1])
    end_milliseconds = int(end_datetime_str.split('.')[-1])

    # Parse the datetime without milliseconds
    start_datetime_obj = datetime.strptime(start_datetime_str[:-7], format_str)
    end_datetime_obj = datetime.strptime(end_datetime_str[:-7], format_str)

    # Add the milliseconds to the datetime object
    start_datetime_obj = start_datetime_obj.replace(microsecond=start_milliseconds)
    end_datetime_obj = end_datetime_obj.replace(microsecond=end_milliseconds)
    
    return start_datetime_obj, end_datetime_obj


def create_timestamps(IMU_start_timestamp, IMU_end_timestamp, sampling_freq):
    # Calculate the time difference between start and end timestamps
    time_diff = IMU_end_timestamp - IMU_start_timestamp
    
    # Calculate the time interval in milliseconds
    interval_ms = int(1 / sampling_freq * 1000)
    
    # Create a range of timestamps with milliseconds spaced by interval_ms
    timestamps = [IMU_start_timestamp + timedelta(milliseconds=interval_ms*i) for i in range(int(time_diff.total_seconds() * sampling_freq) + 1)]
    
    # Convert the timestamps to a DataFrame
    timestamps_df = pd.DataFrame(timestamps, columns=['timestamp'])
    
    return timestamps_df


def plot_x_axis_acceleration(data, sampling_freq):
    # Select only the x-axis acceleration column
    x_acceleration = data['acc_x']

    # Calculate time based on number of samples and sampling frequency
    time = np.arange(len(data)) / sampling_freq

    # Create a larger plot
    plt.figure(figsize=(22, 11))

    # Plot x-axis acceleration over time
    plt.plot(time, x_acceleration)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Acceleration (g)')
    plt.title('X-Axis Acceleration over Time')
    plt.show()


def plot_acceleration_magnitude(data, sampling_freq):
    # Calculate the magnitude of acceleration
    magnitude = np.sqrt(data['acc_x']**2 + data['acc_y']**2 + data['acc_z']**2)

    # Calculate time based on the number of samples and sampling frequency
    time = np.arange(len(data)) / sampling_freq

    # Create a larger plot
    plt.figure(figsize=(22, 11))

    # Plot acceleration magnitude over time
    plt.plot(time, magnitude)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Acceleration Magnitude (g)')
    plt.title('Acceleration Magnitude over Time')
    plt.show()


def convert_to_timedelta(time_in_seconds):
    hours = int(time_in_seconds // 3600)
    minutes = int((time_in_seconds % 3600) // 60)
    seconds = int((time_in_seconds % 60))
    milliseconds = int((time_in_seconds % 1) * 1000)
    return timedelta(hours=hours, minutes=minutes, seconds=seconds, milliseconds=milliseconds)





def trim_data(df, start_duration, end_duration):
    # Convert timepoints to timedelta objects
    start_delta = timedelta(seconds=start_duration)
    end_delta = timedelta(seconds=end_duration)

    # Get the first timestamp as the basis
    basis_timestamp = df['timestamp'].iloc[0]

    # Calculate start and end trimming timestamps
    start_trimming = basis_timestamp + start_delta
    end_trimming = basis_timestamp + end_delta

    # Trim the DataFrame based on the timestamps
    trimmed_df = df[(df['timestamp'] >= start_trimming) & (df['timestamp'] <= end_trimming)]

    return trimmed_df





def copy_video_with_new_name(input_file, output_file):
    if os.path.exists(output_file):
        print(f"File '{output_file}' already exists. Skipping the copy process.")
    else:
        command = ['ffmpeg', '-i', input_file, '-c', 'copy', output_file]
        subprocess.run(command)
        print(f"Video copied: '{input_file}' -> '{output_file}'")


def split_video_into_segments(input_video, output_folder, trimmed_number_frames):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get the duration of the input video using FFprobe
    duration_command = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', input_video]
    duration_output = subprocess.check_output(duration_command).decode('utf-8').strip()
    video_duration = float(duration_output)

    # Calculate the number of segments
    num_segments = int(video_duration / 60) + 1

    # Remove the ".mp4" extension from the input video filename
    input_filename = os.path.splitext(os.path.basename(input_video))[0]

    # Split the video into segments
    segment_frames = 0  # Variable to keep track of total frames in the segments
    for i in range(num_segments):
        start_time = i * 60
        end_time = min((i + 1) * 60, video_duration)

        output_video = os.path.join(output_folder, f"{input_filename}_{i+1}.mp4")

        # Check if the segment already exists
        if os.path.exists(output_video):
            print(f"Segment {i+1} already exists. Skipping...")
        else:
            # Execute the FFmpeg command to split the video and re-encode the segment
            command = ['ffmpeg', '-i', input_video, '-ss', str(start_time), '-to', str(end_time), output_video]
            subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"Segment {i+1} saved: {output_video}")

        # Get the number of frames in the segment
        frames_output, _, _, _, _ = extract_metadata(output_video)
        segment_frames += int(frames_output)

    print("Video splitting completed!")

    # Check if the total number of frames in the segments matches the trimmed_number_frames
    if segment_frames == trimmed_number_frames:
        print("Total number of frames in the segments matches the trimmed number of frames.")
    else:
        raise ValueError("Error: Total number of frames in the segments does not match the trimmed number of frames.")


def plt_zoom(df, duration1):
    """
    Plot a zoomed-in view of the data around the given timestamp.

    Args:
        df (DataFrame): The input DataFrame containing 'time' and 'magnitude' columns.
        duration1 (float): The timestamp to zoom around in seconds.

    Returns:
        None
    """
    # Convert duration1 to seconds
    duration1 = float(duration1)
    
    
    # Convert 'time' column to float representation of seconds
    df['time'] = df['time'].dt.total_seconds()
    
    
    # Define the time range for the zoomed-in view (2 seconds before and after the duration1)
    start_time = duration1 - 5.0
    end_time = duration1 + 5.0

    # Filter the DataFrame based on the time range
    zoom_df = df[(df['time'] >= start_time) & (df['time'] <= end_time)]

    # Plot magnitude over time with the zoomed-in view
    plt.figure(figsize=(12, 6))
    plt.plot(zoom_df['time'], zoom_df['magnitude'], label='Zoom')

    # Find the index of the closest value to duration1 in the 'time' column
    closest_index = np.abs(zoom_df['time'] - duration1).idxmin()

    # Retrieve the corresponding x and y values
    closest_time = zoom_df.loc[closest_index, 'time']
    closest_magnitude = zoom_df.loc[closest_index, 'magnitude']

    # Plot a point at the closest x and y values
    plt.scatter(closest_time, closest_magnitude, color='red', label='Duration1')

    plt.xlabel('Time')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.show()

    
def plot_acceleration_with_timepoints(df, duration1, trimmed_number_frames, trimmed_recording_time):
    sampling_frequency = 50

    # Calculate magnitude of acceleration
    df['magnitude'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)

    # Compute time values based on sampling frequency and round to millisecond
    df['time'] = np.round(pd.Series(range(len(df))) / sampling_frequency, 3)

    # Calculate derivative of magnitude using numpy gradient function
    df['magnitude_derivative'] = np.gradient(df['magnitude'], df['time'])

    duration2 = duration1 + trimmed_recording_time

    # Plot magnitude over time
    plt.figure(figsize=(12, 6))
    plt.plot(df['time'], df['magnitude'], label='Magnitude')

   # Plot special red dots for given durations if provided
    if duration1:
        closest_time = df.loc[np.abs(df['time'] - duration1).idxmin(), 'time']
        closest_magnitude = df.loc[np.abs(df['time'] - duration1).idxmin(), 'magnitude']
        plt.plot(closest_time, closest_magnitude, 'ro', label='Duration 1')

    if duration2:
        closest_time = df.loc[np.abs(df['time'] - duration2).idxmin(), 'time']
        closest_magnitude = df.loc[np.abs(df['time'] - duration2).idxmin(), 'magnitude']
        plt.plot(closest_time, closest_magnitude, 'ro', label='Duration 2')


    plt.xlabel('Time')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.show()

    # Plot magnitude derivative over time
    plt.figure(figsize=(12, 6))
    plt.plot(df['time'], df['magnitude_derivative'], label='Magnitude Derivative')

  # Plot special red dots for given durations if provided
    if duration1:
        closest_time = df.loc[np.abs(df['time'] - duration1).idxmin(), 'time']
        closest_magnitude = df.loc[np.abs(df['time'] - duration1).idxmin(), 'magnitude']
        plt.plot(closest_time, closest_magnitude, 'ro', label='Duration 1')

    if duration2:
        closest_time = df.loc[np.abs(df['time'] - duration2).idxmin(), 'time']
        closest_magnitude = df.loc[np.abs(df['time'] - duration2).idxmin(), 'magnitude']
        plt.plot(closest_time, closest_magnitude, 'ro', label='Duration 2')
        
    plt.xlabel('Time')
    plt.ylabel('Magnitude Derivative')
    plt.legend()
    plt.show()
    
    # Calculate time duration and sample count between the two points
    duration = abs(duration2 - duration1)
    sample_count = len(df[(df['time'] >= duration1) & (df['time'] <= duration2)])

    # Adjust sample_count to ensure sample_count = 2 * trimmed_number_frames
    difference = sample_count - 2 * trimmed_number_frames
    if difference > 0:
        duration2 -= difference / sampling_frequency
        sample_count -= difference

    print("Duration 1:", duration1)
    print("Duration 2:", duration2)
    print("Number of Samples:", sample_count)
    print("Number of Video Frames:", trimmed_number_frames)
    if trimmed_number_frames * 2 == sample_count:
        print("Conditions ok")

    # Reorder columns with 'time' as the first feature
    cols = ['time'] + [col for col in df.columns if col != 'time']
    df = df[cols]
    
    # Apply the conversion to the 'time' column
    df['time'] = df['time'].apply(convert_to_timedelta)
    
    # Zoom in to adjust duration1
    plt_zoom(df, duration1)
    
    # Remove these features
    df = df.drop(['magnitude_derivative', 'magnitude'], axis=1)
    
    return df, duration1, duration2


def trim_data(df, start_duration, end_duration):
    """
    Trim a DataFrame based on start and end durations.

    Args:
        df (DataFrame): The input DataFrame containing a 'timestamp' column.
        start_duration (float): Start duration in seconds, precise up to milliseconds.
        end_duration (float): End duration in seconds, precise up to milliseconds.

    Returns:
        DataFrame: Trimmed DataFrame based on the start and end durations.
    """
    # Convert start and end durations to milliseconds
    start_ms = int(start_duration * 1000)
    end_ms = int(end_duration * 1000)
    
    # Get the first timestamp as the basis
    basis_timestamp = df['timestamp'].iloc[0]

    # Calculate start and end trimming timestamps with milliseconds
    start_trimming = basis_timestamp + timedelta(milliseconds=start_ms)
    end_trimming = basis_timestamp + timedelta(milliseconds=end_ms)
    
    # Trim the DataFrame based on the timestamps
    trimmed_df = df[(df['timestamp'] >= start_trimming) & (df['timestamp'] <= end_trimming)]

    return trimmed_df


def save_integers_to_file(path, participant_id, int1, int2, int3):
    # Create the output folder if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)

    # Specify the file name
    file_name = f'{participant_id}_video_parameters.txt'
    file_path = os.path.join(path, file_name)

    # Save the integers to the file
    with open(file_path, 'w') as file:
        file.write(f'Frame Start: {int1}\n')
        file.write(f'Frame End: {int2}\n')
        file.write(f'Duration: {int3}\n')

    print(f"The integers have been saved to the file: '{file_path}'")


def save_data(path, participant_id, frame_start, frame_end, duration1, LW_trimmed_data, RW_trimmed_data, chest_trimmed_data):
    """
    Save video parameters and trimmed data to CSV files.

    Args:
        path (str): Path to the folder for saving the files.
        participant_id (str): Participant ID.
        frame_start (int): Start frame of the video.
        frame_end (int): End frame of the video.
        duration1 (float): Duration of the video.
        LW_trimmed_data (DataFrame): Trimmed data for LW.
        RW_trimmed_data (DataFrame): Trimmed data for RW.
        chest_trimmed_data (DataFrame): Trimmed data for chest.

    Returns:
        None
    """
    # Save video parameters to a text file
    save_integers_to_file(path, participant_id, frame_start, frame_end, duration1)
    print("Video parameters saved.")

    # Specify the output CSV file names
    lw_output_filename = 'trimmed_LW_data.csv'
    rw_output_filename = 'trimmed_RW_data.csv'
    chest_output_filename = 'trimmed_chest_data.csv'

    # Construct the full file paths for the output CSV files
    lw_output_path = os.path.join(path, lw_output_filename)
    rw_output_path = os.path.join(path, rw_output_filename)
    chest_output_path = os.path.join(path, chest_output_filename)

    # Save the trimmed data
    LW_trimmed_data.to_csv(lw_output_path, index=False)
    print("Trimmed data for LW saved to:", lw_output_path)
    
    RW_trimmed_data.to_csv(rw_output_path, index=False)
    print("Trimmed data for RW saved to:", rw_output_path)
    
    chest_trimmed_data.to_csv(chest_output_path, index=False)
    print("Trimmed data for chest saved to:", chest_output_path)


def plt_zoom(df, duration1):
    """
    Plot a zoomed-in view of the data around the given timestamp.

    Args:
        df (DataFrame): The input DataFrame containing 'time' and 'magnitude' columns.
        duration1 (float): The timestamp to zoom around in seconds.

    Returns:
        None
    """
    # Convert duration1 to seconds
    duration1 = float(duration1)

    # Create a copy of the DataFrame to avoid direct changes
    df_copy = df.copy()

    # Convert 'time' column to float representation of seconds in the copied DataFrame
    df_copy['time'] = df_copy['time'].dt.total_seconds()

    # Define the time range for the zoomed-in view (2 seconds before and after the duration1)
    start_time = duration1 - 5.0
    end_time = duration1 + 5.0

    # Filter the copied DataFrame based on the time range
    zoom_df = df_copy[(df_copy['time'] >= start_time) & (df_copy['time'] <= end_time)]

    # Plot magnitude over time with the zoomed-in view
    plt.figure(figsize=(12, 6))
    plt.plot(zoom_df['time'], zoom_df['magnitude'], label='Zoom')

    # Find the index of the closest value to duration1 in the 'time' column
    closest_index = np.abs(zoom_df['time'] - duration1).idxmin()

    # Retrieve the corresponding x and y values
    closest_time = zoom_df.loc[closest_index, 'time']
    closest_magnitude = zoom_df.loc[closest_index, 'magnitude']

    # Plot a point at the closest x and y values
    plt.scatter(closest_time, closest_magnitude, color='red', label='Duration1')

    plt.xlabel('Time')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.show()


def plot_acceleration_with_timepoints(df, duration1, trimmed_number_frames, trimmed_recording_time):
    sampling_frequency = 50

    # Create a copy of the DataFrame to avoid direct changes
    df_copy = df.copy()

    # Calculate magnitude of acceleration in the copied DataFrame
    df_copy['magnitude'] = np.sqrt(df_copy['acc_x']**2 + df_copy['acc_y']**2 + df_copy['acc_z']**2)

    # Compute time values based on sampling frequency and round to millisecond in the copied DataFrame
    df_copy['time'] = np.round(pd.Series(range(len(df_copy))) / sampling_frequency, 3)

    # Calculate derivative of magnitude using numpy gradient function in the copied DataFrame
    df_copy['magnitude_derivative'] = np.gradient(df_copy['magnitude'], df_copy['time'])

    duration2 = duration1 + trimmed_recording_time

    # Plot magnitude over time
    plt.figure(figsize=(12, 6))
    plt.plot(df_copy['time'], df_copy['magnitude'], label='Magnitude')

    # Plot special red dots for given durations if provided
    if duration1:
        closest_time = df_copy.loc[np.abs(df_copy['time'] - duration1).idxmin(), 'time']
        closest_magnitude = df_copy.loc[np.abs(df_copy['time'] - duration1).idxmin(), 'magnitude']
        plt.plot(closest_time, closest_magnitude, 'ro', label='Duration 1')

    if duration2:
        closest_time = df_copy.loc[np.abs(df_copy['time'] - duration2).idxmin(), 'time']
        closest_magnitude = df_copy.loc[np.abs(df_copy['time'] - duration2).idxmin(), 'magnitude']
        plt.plot(closest_time, closest_magnitude, 'ro', label='Duration 2')

    plt.xlabel('Time')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.show()

    # Plot magnitude derivative over time
    plt.figure(figsize=(12, 6))
    plt.plot(df_copy['time'], df_copy['magnitude_derivative'], label='Magnitude Derivative')

    # Plot special red dots for given durations if provided
    if duration1:
        closest_time = df_copy.loc[np.abs(df_copy['time'] - duration1).idxmin(), 'time']
        closest_magnitude = df_copy.loc[np.abs(df_copy['time'] - duration1).idxmin(), 'magnitude']
        plt.plot(closest_time, closest_magnitude, 'ro', label='Duration 1')

    if duration2:
        closest_time = df_copy.loc[np.abs(df_copy['time'] - duration2).idxmin(), 'time']
        closest_magnitude = df_copy.loc[np.abs(df_copy['time'] - duration2).idxmin(), 'magnitude']
        plt.plot(closest_time, closest_magnitude, 'ro', label='Duration 2')

    plt.xlabel('Time')
    plt.ylabel('Magnitude Derivative')
    plt.legend()
    plt.show()

    # Calculate time duration and sample count between the two points
    duration = abs(duration2 - duration1)
    sample_count = len(df_copy[(df_copy['time'] >= duration1) & (df_copy['time'] <= duration2)])

    # Adjust sample_count to ensure sample_count = 2 * trimmed_number_frames
    difference = sample_count - 2 * trimmed_number_frames
    if difference > 0:
        duration2 -= difference / sampling_frequency
        sample_count -= difference

    print("Duration 1:", duration1)
    print("Duration 2:", duration2)
    print("Number of Samples:", sample_count)
    print("Number of Video Frames:", trimmed_number_frames)
    if trimmed_number_frames * 2 == sample_count:
        print("Conditions ok")

    # Reorder columns with 'time' as the first feature in the copied DataFrame
    cols = ['time'] + [col for col in df_copy.columns if col != 'time']
    df_copy = df_copy[cols]

    # Apply the conversion to the 'time' column in the copied DataFrame
    df_copy['time'] = df_copy['time'].apply(convert_to_timedelta)

    # Zoom in to adjust duration1 using the copied DataFrame
    plt_zoom(df_copy, duration1)

    # Zoom in to adjust duration2 using the copied DataFrame
    plt_zoom(df_copy, duration2)

    # Remove these features from the copied DataFrame
    df_copy = df_copy.drop(['magnitude_derivative', 'magnitude'], axis=1)

    return df_copy, duration1, duration2
