#calculate inter-rater consensus for the 3 raters

from utilities import *
from functions.extract_mask_from_video import *
from sklearn.metrics import cohen_kappa_score

def read_ndjson(file_path='../data/CreateStudy/consensus_labels.ndjson'):
    df = pd.read_json(file_path, lines=True)
    return df

def parse_labler(label_data):
    functional_NF_frame_array, functional_NF_label_array = [], []
    functional_primitive_frame_array, functional_primitive_label_array = [], []
    task_frame_array, task_label_array = [], []
    
    for key, value in label_data['annotations']['frames'].items():
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
    
    functional_primitive_per_frame_one = fill_mask(functional_primitive_frame_array, functional_primitive_label_array, ['reach', 'reposition', 'transport', 'gesture', 'idle', 'stabilization', 'arm_not_visible'])
    functional_NF_per_frame_one = fill_mask(functional_NF_frame_array, functional_NF_label_array, ['functional_movement', 'non_functional_movement', 'arm_not_visible'])
    task_per_frame_one = fill_mask(task_frame_array, task_label_array, ['open_bottle_and_pour_glass', 'drink', 'fold_rags_towels', 'sort_documents', 'brooming', 'putting_on_and_off_coat', 'keyboard_typing', 'stapling', 'walking', 'open_and_close_door', 'resting', 'other', 'wipe_table', 'light_switch'])

    return functional_primitive_per_frame_one, functional_NF_per_frame_one, task_per_frame_one
        
def parse_json():
    FILE_PATH = '../data/CreateStudy/consensus_labels.ndjson'
    project_key = 'clw8u6yxb02dk07yd2jqg4vgk'
    # Initialize an empty list to store JSON objects
    data_list = []
    # Open the NDJSON file
    with open(FILE_PATH, 'r') as file:
        for line in file:
            # Parse the JSON object from the line and add it to the list
            json_obj = json.loads(line.strip())  # strip() removes any extra whitespace or newline characters
            data_list.append(json_obj)

    labels_per_datarow = []
    for data_row in data_list:
        data_row_dict = {}

        labler_zero = data_row['projects'][project_key]['labels'][0]['label_details']['created_by']
        label_zero_data = data_row['projects'][project_key]['labels'][0]
        functional_primitive_per_frame_zero, functional_NF_per_frame_zero, task_per_frame_zero = parse_labler(label_zero_data)
        labelbox_consensus_zero = data_row['projects'][project_key]['labels'][0]['performance_details']['consensus_score']
        labler_zero_dict = {'functional_primitive_per_frame': functional_primitive_per_frame_zero, 'functional_NF_per_frame': functional_NF_per_frame_zero, 'task_per_frame': task_per_frame_zero, 'labelbox_consensus': labelbox_consensus_zero, 'labler': labler_zero}
        
        labler_one = data_row['projects'][project_key]['labels'][1]['label_details']['created_by']
        label_one_data = data_row['projects'][project_key]['labels'][1]
        functional_primitive_per_frame_one, functional_NF_per_frame_one, task_per_frame_one = parse_labler(label_one_data)
        labelbox_consensus_one = data_row['projects'][project_key]['labels'][1]['performance_details']['consensus_score']
        labler_one_dict = {'functional_primitive_per_frame': functional_primitive_per_frame_one, 'functional_NF_per_frame': functional_NF_per_frame_one, 'task_per_frame': task_per_frame_one, 'labelbox_consensus': labelbox_consensus_one, 'labler': labler_one}

        data_row_dict['labler_zero'] = labler_zero_dict
        data_row_dict['labler_one'] = labler_one_dict
        data_row_dict['video_id'] = data_row['data_row']['external_id']
        labels_per_datarow.append(data_row_dict)

    return labels_per_datarow

def calculate_cohens_k(label_zero, label_one, labels):
    #TODO something is wrong here
    return cohen_kappa_score(label_zero, label_one, labels=labels)

def calculate_labelbox_consensus(label_zero, label_one, labels):
    # Count elements that are equal at the same position
    count = sum(1 for a, b in zip(label_zero, label_one) if a == b)
    # Calculate the consensus score
    return count / len(label_zero)

def calculate_consensus_score(labels_per_datarow):
    #Labelbox consensus, Gwet's AC1 agreement score were 0.91 +- 0.02 and 0.94 +- 0.02 respectively \cite{subash2022comparing}. Cohen's coefficient K>= 0.96 \cite{parnandi2022primseq}.

    # Initialize the consensus scores
    functional_primitive_consensus = []
    functional_NF_consensus = []
    task_consensus = []
    # Iterate over the list of labels per data row
    for data_row in labels_per_datarow:
        # Get the labels for the two lablers
        #print(data_row['video_id'])
        labler_zero = data_row['labler_zero']
        labler_one = data_row['labler_one']
        # Calculate the consensus scores for the functional movement primitives
        functional_primitive_consensus_score = calculate_labelbox_consensus(labler_zero['functional_primitive_per_frame'], labler_one['functional_primitive_per_frame'], labels=['reach', 'reposition', 'transport', 'gesture', 'idle', 'stabilization', 'arm_not_visible'])
        functional_primitive_consensus.append(functional_primitive_consensus_score)
        # Calculate the consensus scores for the functional / non-functional movement
        functional_NF_consensus_score = calculate_labelbox_consensus(labler_zero['functional_NF_per_frame'], labler_one['functional_NF_per_frame'], labels=['functional_movement', 'non_functional_movement', 'arm_not_visible'])
        functional_NF_consensus.append(functional_NF_consensus_score)
        # Calculate the consensus scores for the task
        if 'RW' in data_row['video_id']:
            continue #RW videos don't have task lables
        task_consensus_score = calculate_labelbox_consensus(labler_zero['task_per_frame'], labler_one['task_per_frame'], ['open_bottle_and_pour_glass', 'drink', 'fold_rags_towels', 'sort_documents', 'brooming', 'putting_on_and_off_coat', 'keyboard_typing', 'stapling', 'walking', 'open_and_close_door', 'resting', 'other', 'wipe_table', 'light_switch'])
        task_consensus.append(task_consensus_score)
    
    return functional_primitive_consensus, functional_NF_consensus, task_consensus

def main():
    return 0

if __name__ == "__main__":
    main()