import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import labelbox
import numpy as np
import os


def extract_json_data(api_key, project_id):
    """
    Extracts JSON data by exporting a project using the Labelbox API.

    Args:
        api_key (str): The Labelbox API key.
        project_id (str): The ID of the project to export.

    Returns:
        dict: The exported JSON data.
    """
    try:
        # Create a Labelbox client object using the API key
        client = labelbox.Client(api_key=api_key)

        # Retrieve the project using the project ID
        project = client.get_project(project_id)

        # Set export parameters (if needed)
        export_params = {}

        # Initiate the export task for the project using the export parameters
        export_task = project.export_v2(params=export_params)

        # Wait for the export task to complete
        export_task.wait_till_done()

        # Check if there are any errors in the export task
        if export_task.errors:
            print(export_task.errors)
        else:
            print("API connection to Labelbox successful.")

        # Get the result of the export task (exported JSON data)
        export_json = export_task.result

        # Return the exported JSON data
        return export_json

    except Exception as e:
        print("API connection to Labelbox failed:", str(e))
        return None


def get_video_path(folder, side, video_number, number_videos):
    # Ensure video_number is less than or equal to number_videos
    if video_number > number_videos:
        raise ValueError("Invalid video number. Please choose a valid video number.")

    # Convert the video_number into a string
    video_number_str = str(video_number)

    # Create the path
    video_path = f'{folder}//splitted_videos_{side}//{folder}_{side}_{video_number_str}.mp4'
    return video_path


def get_video_path_labelbox(folder, side, video_number, number_videos):
    video_path = get_video_path(folder, side, video_number, number_videos)
    video_path_labelbox = os.path.basename(video_path)
    return video_path_labelbox


def get_all_video_path_participant(folder, side, number_videos):
    video_paths_LW = []
    video_paths_RW = []

    # Iterate over all sides
    for s in side:
        # Iterate over all videos
        for video_number in range(1, number_videos + 1):
            video_path = get_video_path(folder, s, video_number, number_videos)
            if s == 'LW':
                video_paths_LW.append(video_path)
            elif s == 'RW':
                video_paths_RW.append(video_path)

    return [video_paths_LW, video_paths_RW]


def get_all_video_path_participant_labelbox(participant_id, number_videos): 
    videos_paths_LW = []
    videos_paths_RW = []
    
    for i in range(1, number_videos + 1): 
        video_path_LW = participant_id + '_LW_' + str(i) + '.mp4'
        video_path_RW = participant_id + '_RW_' + str(i) + '.mp4'
        
        videos_paths_LW.append(video_path_LW)
        videos_paths_RW.append(video_path_RW)
    
    return videos_paths_LW, videos_paths_RW


def get_folder_element_count(path):
    """
    Get the number of elements (files and subfolders) in a folder.

    Args:
        path (str): Path to the folder.

    Returns:
        int: Number of elements in the folder.
    """
    # Check if the path is a valid directory
    if not os.path.isdir(path):
        raise ValueError(f"{path} is not a valid directory.")

    # Get the list of files and subfolders in the directory
    elements = os.listdir(path)

    # Count the number of elements
    count = len(elements)

    return count


def segment_data(data, external_id):
    """
    Segments data based on a specific sub-project ID.

    Args:
        data (list): The full json data as a list of dictionaries.
        id (str): The specified part of the JSON data.

    Returns:
        list: Segmented JSON data.
    """
    segmented_data = []
    current_segment = []

    for item in data:
        if 'data_row' in item:
            data_row = item['data_row']
            if 'external_id' in data_row and data_row['external_id'] == external_id:
                if current_segment:
                    segmented_data.append(current_segment)
                    current_segment = []
                current_segment.append(item)
        else:
            current_segment.append(item)

    if current_segment:
        segmented_data.append(current_segment)
    
    # Convert it to a JSON-formatted string
    segmented_data = json.dumps(segmented_data)

    return segmented_data


def closest_label(array, numbers_of_interest):
    modified_array = array.copy()

    for i in range(len(modified_array)):
        if modified_array[i] == -5:
            closest_index = None
            min_distance = float('inf')

            for j in range(len(modified_array)):
                if modified_array[j] in numbers_of_interest:
                    distance = abs(j - i)
                    if distance < min_distance:
                        closest_index = j
                        min_distance = distance

            if closest_index is not None:
                modified_array[i] = modified_array[closest_index]

    return modified_array


def fill_mask(frame_array, label_array, mask_labels):

    # Prepare a mask array of the size of the maximum value contained inside frame_array
    mask_size = max(frame_array, default=0)
    mask = [-5] * mask_size

    # Place the correct value of label at the correct frame position
    for frame, label in zip(frame_array, label_array):
        mask[frame - 1] = label

    # Fill the gaps between frames 
    mask = closest_label(mask, mask_labels)
    mask = np.array(mask)

    return mask


def get_label_mask(segmented_data, project_key='clw8u6yxb02dk07yd2jqg4vgk'): #replaces get_mask, parses JSON exported from labelbox

    segmented_data = json.loads(segmented_data) #conver JSON string to Python dictionary/list
    labeled_frames = segmented_data[0][0]['projects'][project_key]['labels'][0]['annotations']['frames']
    
    functional_NF_frame_array, functional_NF_label_array = [], []
    functional_primitive_frame_array, functional_primitive_label_array = [], []
    task_frame_array, task_label_array = [], []

    for key, value in labeled_frames.items():
        for schema in value['classifications']:
            if schema['name'] == 'functional movement primitive' or schema['name'] == 'exclusion':
                functional_primitive_frame_array.append(int(key))
                functional_primitive_label_array.append(schema['radio_answer']['value'])
            if schema['name'] == 'functional / non-functional' or schema['name'] == 'exclusion':
                functional_NF_frame_array.append(int(key))
                functional_NF_label_array.append(schema['radio_answer']['value'])
            if schema['name'] == 'task':
                task_frame_array.append(int(key))
                task_label_array.append(schema['radio_answer']['value'])
    
    if len(functional_primitive_frame_array) != len(functional_primitive_label_array) or len(functional_NF_frame_array) != len(functional_NF_label_array) or len(task_frame_array) != len(task_label_array):
        raise ValueError("Frame and label arrays should have the same size.")

    functional_primitive_per_frame = fill_mask(functional_primitive_frame_array, functional_primitive_label_array, ['reach', 'reposition', 'transport', 'gesture', 'idle', 'stabilization', 'arm_not_visible'])
    functional_NF_per_frame = fill_mask(functional_NF_frame_array, functional_NF_label_array, ['functional_movement', 'non_functional_movement', 'arm_not_visible'])
    task_per_frame = fill_mask(task_frame_array, task_label_array, ['open_bottle_and_pour_glass', 'drink', 'fold_rags_towels', 'sort_documents', 'brooming', 'putting_on_and_off_coat', 'keyboard_typing', 'stapling', 'walking', 'open_and_close_door', 'resting', 'other', 'wipe_table', 'light_switch'])

    return functional_primitive_per_frame, functional_NF_per_frame, task_per_frame


def convert_labels_to_int(label_array):
    label_to_int = {'functional_movement': 1, 'non_functional_movement': 0, 'reach': 2, 'reposition': 3, 'transport': 4, 'gesture': 5, 'idle': 6, 'stabilization': 7, 'arm_not_visible': 999}
    int_array = [label_to_int[label] for label in label_array]
    return int_array


def extract_mask_from_videos(videos_paths, export_json, project_key='clw8u6yxb02dk07yd2jqg4vgk'):
    mask_per_video = []
    primitives_per_video = []
    tasks_per_video = []

    # Loop over all the video_path in the videos_paths array
    for video_path in videos_paths:
        # Call segment_data function
        segmented_data = segment_data(export_json, video_path)
        
        # Extract the masks using the labeled frames
        #mask_video = get_mask(segmented_data)
        primitives_per_frame, mask_video, tasks_per_frame = get_label_mask(segmented_data, project_key)
        mask_video = convert_labels_to_int(mask_video)
        primitives_per_frame = convert_labels_to_int(primitives_per_frame)
        mask_per_video.append(mask_video)
        primitives_per_video.append(primitives_per_frame)
        tasks_per_video.append(tasks_per_frame)

    # Merge all the mask_video together using np.concatenate
    mask = np.concatenate(mask_per_video)
    primitives = np.concatenate(primitives_per_video)
    tasks = np.concatenate(tasks_per_video)

    return primitives, mask, tasks


def plot_movement_tendency(data, frame_rate = 25):
    """
    Plot the movement tendency over time and display the percentage of occurrence.

    Args:
        data (DataFrame or ndarray): Input DataFrame with a 'mask' column containing values -1 (NA), 0 (NF), or 1 (F),
                                     or a single-column numpy array representing the mask data.

    Returns:
        None (displays line and bar plots).
    """

    # If input is a DataFrame, extract the 'mask' column
    if isinstance(data, pd.DataFrame):
        mask_data = data['mask'].values
    # If input is a numpy array, use it as the mask data
    elif isinstance(data, np.ndarray):
        mask_data = data.flatten()
    else:
        raise ValueError("Invalid input type. Input should be a DataFrame or a numpy array.")

    # Delete entries in mask that are 999 (labeled exclusion)
    mask_data = mask_data[mask_data != 999]
    
    # Compute time based on the frame rate (FPS)
    time = np.arange(len(mask_data)) / frame_rate

    # Calculate the percentage of occurrence for each element in the mask data
    unique_elements, counts = np.unique(mask_data, return_counts=True)
    total_count = len(mask_data)
    percentages = counts / total_count * 100

    # Map the original values to labels for plotting
    label_mapping = {-1: "WBM", 1: "Functional Movement", 0: "Non-functional Movement"}  # Include -1 in label mapping
    unique_labels = [label_mapping[element] for element in unique_elements]

    # Plot the bar plot with increased size
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(unique_labels)), percentages)
    plt.xlabel('Elements')
    plt.ylabel('Percentage')
    plt.title('Percentage of Occurrence')
    plt.xticks(range(len(unique_labels)), unique_labels)

    # Add percentage values on top of each bar
    for i, v in enumerate(percentages):
        plt.text(i, v + 1, f'{v:.1f}%', ha='center')

    # Add threshold line at 20%
    plt.axhline(y=20, color='red', linestyle='--')
    plt.text(len(unique_labels) - 1, 20, 'Imbalanced Data Warning', color='red', ha='right')

    # Adjust the limits of the y-axis
    plt.ylim(0, max(percentages) + 10)
    plt.show()


def save_masks_as_csv(GT_mask_LW, GT_mask_RW, folder):
    """
    Save masks as CSV files.

    Args:
        GT_mask_LW (np.ndarray): Numpy array containing the left-hand masks.
        GT_mask_RW (np.ndarray): Numpy array containing the right-hand masks.
        folder (str): Folder path to save the CSV files.

    Returns:
        None.
    """
    # Convert the numpy arrays to pandas DataFrames
    GT_mask_LW_df = pd.DataFrame(GT_mask_LW, columns=['mask'])
    GT_mask_RW_df = pd.DataFrame(GT_mask_RW, columns=['mask'])

    # Specify the output CSV file names
    lw_output_filename = 'GT_mask_LW.csv'
    rw_output_filename = 'GT_mask_RW.csv'

    # Construct the full file paths for the output CSV files
    lw_output_path = os.path.join(folder, lw_output_filename)
    rw_output_path = os.path.join(folder, rw_output_filename)

    # Save the trimmed data
    GT_mask_LW_df.to_csv(lw_output_path, index=False)
    GT_mask_RW_df.to_csv(rw_output_path, index=False)