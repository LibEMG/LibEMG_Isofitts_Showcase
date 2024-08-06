import pickle
import numpy as np
import math
import matplotlib.pyplot as plt
from os import walk
from libemg.emg_predictor import EMGClassifier
from libemg.feature_extractor import FeatureExtractor
from libemg.utils import make_regex
from libemg.data_handler import OfflineDataHandler, RegexFilter
from libemg.offline_metrics import OfflineMetrics

def evaluate_offline_data():
    WINDOW_SIZE = 40
    WINDOW_INCREMENT = 20

    offline_metrics = {
        'classifier': [],
        'metrics': [],
    }

    dataset_folder = 'data/'
    regex_filters = [
        RegexFilter(left_bound = "C_", right_bound="_R", values = ["0","1","2","3","4"], description='classes'),
        RegexFilter(left_bound = "R_", right_bound="_emg.csv", values = ["0", "1", "2"], description='reps'),
    ]

    odh = OfflineDataHandler()
    odh.get_data(dataset_folder, regex_filters, delimiter=",")

    fe = FeatureExtractor()

    train_odh = odh.isolate_data(key="reps", values=[0,1])
    train_windows, train_metadata = train_odh.parse_windows(WINDOW_SIZE,WINDOW_INCREMENT)
    test_odh = odh.isolate_data(key="reps", values=[2])
    test_windows, test_metadata = test_odh.parse_windows(WINDOW_SIZE,WINDOW_INCREMENT)

    data_set = {}
    data_set['testing_features'] = fe.extract_feature_group('HTD', test_windows)
    data_set['training_features'] = fe.extract_feature_group('HTD', train_windows)
    data_set['testing_labels'] = test_metadata['classes']
    data_set['training_labels'] = train_metadata['classes']

    om = OfflineMetrics()
    metrics = ['CA', 'AER', 'INS', 'CONF_MAT']
    # Normal Case - Test all different classifiers
    for model in ['LDA', 'SVM', 'KNN', 'RF']:
        classifier = EMGClassifier(model)
        classifier.fit(data_set.copy())
        preds, probs = classifier.run(data_set['testing_features'])
        out_metrics = om.extract_offline_metrics(metrics, data_set['testing_labels'], preds, 2)
        offline_metrics['classifier'].append(model)
        offline_metrics['metrics'].append(out_metrics)
    return offline_metrics

def read_pickle(location):
    with open(location, 'rb') as f:
        data = pickle.load(f)
    return data

def calculate_overshoots(data):
    overshoots = 0
    trials = np.unique(data['trial_number'])
    for t in trials:
        t_idxs = np.where(data['trial_number'] == t)[0]
        cursor_locs = np.array(data['cursor_position'])[t_idxs]
        targets = np.array(data['goal_circle'])[t_idxs]
        in_bounds = [in_circle(cursor_locs[i], targets[i]) for i in range(0,len(cursor_locs))]
        for i in range(1,len(in_bounds)):
            if in_bounds[i-1] == True and in_bounds[i] == False:
                overshoots+=1 
    return overshoots

def in_circle(cursor, circle):
    return math.sqrt((circle[0] - cursor[0])**2 + (circle[1] - cursor[1])**2) < circle[2]/2 + cursor[2]/2

def calculate_efficiency(data):
    efficiency = []
    trials = np.unique(data['trial_number'])
    for t in trials:
        t_idxs = np.where(data['trial_number'] == t)[0]
        distance_travelled = np.sum([math.dist(data['cursor_position'][t_idxs[i]][0:2], data['cursor_position'][t_idxs[i-1]][0:2]) for i in range(1,len(t_idxs))])
        fastest_path = math.dist((data['cursor_position'][t_idxs[0]])[0:2], (data['goal_circle'][t_idxs[0]])[0:2])
        efficiency.append(fastest_path/distance_travelled)
    return np.mean(efficiency)

def calculate_throughput(data):
    throughput = []
    trials = np.unique(data['trial_number'])
    for t in trials:
        t_idxs = np.where(data['trial_number'] == t)[0]
        distance = math.dist((data['cursor_position'][t_idxs[0]])[0:2], (data['goal_circle'][t_idxs[0]])[0:2])
        width = (data['goal_circle'][t_idxs[0]])[2]
        id = math.log2(distance/width + 1) 
        time = data['global_clock'][t_idxs[-1]] - data['global_clock'][t_idxs[0]]
        throughput.append(id/time)
    return np.mean(throughput)

def extract_fitts_metrics(data):
    fitts_results = {
        'overshoots': [],
        'throughput': [],
        'efficiency': [],
    }
    fitts_results['overshoots'] = calculate_overshoots(data)
    fitts_results['efficiency'] = calculate_efficiency(data)
    fitts_results['throughput'] = calculate_throughput(data)
    return fitts_results


def evaluate_fitts_data():
    path = 'results/'
    filenames = next(walk(path), (None, None, []))[2]
    fitts_metrics = {
        'classifier': [],
        'metrics': [],
    }
    
    for file in filenames:
        data = read_pickle(path + file)
        fitts_metrics['classifier'].append(file.split('_')[1])
        fitts_metrics['metrics'].append(extract_fitts_metrics(data))

    return fitts_metrics

if __name__ == "__main__":
    offline_metrics = evaluate_offline_data()
    fitts_metrics = evaluate_fitts_data()

    num_models = len(fitts_metrics['metrics'][0])

    # Plot bar chart for each classifier - lets look at CA, AER and INS for each classifier
    o_mets = ['CA', 'AER', 'INS']
    f_mets = ['throughput', 'efficiency', 'overshoots']
    fig, axs = plt.subplots(num_models, 2)
    for i in range(0, 3):
        # Plot CA, AER and INS
        x = [x for x in offline_metrics['classifier']]
        y = [(offline_metrics['metrics'][y])[o_mets[i]] for y in range(0, len(offline_metrics['classifier']))]
        axs[i, 0].bar(x,y)
        axs[i, 0].set_title(o_mets[i])
        
        # Plot throughput, efficiency and overshoots 
        x = [x for x in fitts_metrics['classifier']]
        y = [(fitts_metrics['metrics'][y])[f_mets[i]] for y in range(0, len(fitts_metrics['classifier']))]
        axs[i, 1].bar(x,y)
        axs[i, 1].set_title(f_mets[i])
    
    plt.tight_layout()
    plt.show()

    print(offline_metrics)
    print(fitts_metrics)
