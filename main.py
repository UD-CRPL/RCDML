import parser
import data_preprocess as dp
import feature_selection as fs
import validation as val
import pandas as pd
import numpy as np
import classification
import itertools
import sys
from sklearn.model_selection import train_test_split
from datetime import datetime

# Handles the verbosity of the debug mode
def debug_mode(debug):
    if debug == 0:
        dict = {'dataset_split':0,'feature_selection':0,'classification':0, 'validation':0, 'feature_counter':0}
    elif debug == 1:
        dict = {'dataset_split':1,'feature_selection':0,'classification':0, 'validation':1, 'feature_counter':0}
    elif debug == 2:
        dict = {'dataset_split':1,'feature_selection':1,'classification':0, 'validation':1, 'feature_counter':0}
    elif debug == 3:
        dict = {'dataset_split':1,'feature_selection':1,'classification':1, 'validation':1, 'feature_counter':0}
    elif debug == 4:
        dict = {'dataset_split':1,'feature_selection':1,'classification':1, 'validation':1, 'feature_counter':1}
    else:
        sys.exit("ERROR: Unrecognized debugging level. Debug levels available: no debug - 0, validation_mode - 1, +feature_selection - 2, +classification - 3, +feature_counter - 4")
    return dict

def save_feature_counter(result_path, feature_selection, classifier, date, validation, feature_counter, feature_size, debug_mode):
    feature_counter_path = result_path + date + "/" + validation + "/" + feature_selection + "/" + classifier
    # DEBUG MODE
    if debug_mode:
        print("FEATURE COUNTER STARTS")
    # Sorts and selects the top  (feature_size) most frequent features, isn't used in the code right now but keeping this here just in case
    #selected_features = dict(itertools.islice(feature_counter.items(), len(feature_counter) - (1 + feature_size), len(feature_counter) - 1))
    #selected_features = selected_features.keys()
    dp.make_result_dir(feature_counter_path)
    # Converts dictionary into dataframe
    feature_counter_df = pd.DataFrame(feature_counter.items(), columns=["FEATURE", "FREQUENCY"])
    #DEBUG MODE
    if debug_mode:
        debug_path = feature_counter_path + "/debug/"
        # Saves unsorted feature counter
        feature_counter_df.to_csv(debug_path + "/feature_counter.tsv", index = False, sep="\t")
    # sorts the features of feature counter in ascending order
    feature_counter_df = feature_counter_df.sort_values(by=['FREQUENCY'], ascending=False)
    feature_counter_df.to_csv(feature_counter_path + "/feature_counter.tsv", index = False, sep="\t")
    print("FEATURE COUNTER SAVED")
    return

def main():

    parameters = parser.get_parser()

    date = datetime.today().strftime('%m-%d-%Y')
    result_path = parameters['result_path']
    dataset_path = parameters['dataset_path']
    run_name = parameters['run_name']
    project = parameters['project']

    feature_selection = parameters['feature_selection'].split(',')

    feature_size = int(parameters['feature_size'])

    classifiers = parameters['classifiers'].split(',')
    train_set_split = float(parameters['train_test_split'])
    train_test_seed = 200

    validation = parameters['validation']
    iterations = int(parameters['validation_iterations'])
    normalization = parameters['normalization']

    fs_keys = ['dge_path', 'swapped_label', 'drug_feature_path', 'swapped_path']
    feature_selection_parameters = {key: parameters[key] for key in fs_keys}

    save_fc = int(parameters['feature_counter'])

    debug = int(parameters['debug'])

    debug = debug_mode(debug)

    drug_name = parameters['drug_name']

    result_path = result_path + project + "/" + run_name + "/" + drug_name +  "/"

    print("PIPELINE STARTS")

    dataset, dataset_samples = dp.load_dataset(dataset_path, project, normalization)

    print("FINISHED LOADING DATASET")

    labels = dp.load_labels(dataset_path, project, drug_name)

    print("FINISHED LOADING LABELS")

    dataset, labels, samples = dp.sample_match(dataset, labels, dataset_samples)

    feature_counter = fs.build_feature_counter(dataset)

    print("SPLITTING DATASET BASED ON VALIDATION STYLE: " + validation)

    datasets, iterations = val.split_dataset(validation, dataset, labels, train_set_split, iterations)

    for i in feature_selection:
        for j in classifiers:
            for k in range(0, iterations):
                dp.make_result_dir(result_path + date + "/" + validation + "/" + i + "/" + j + "/" + str(k) + "/")

    print("PERFORMING FEATURE SELECTION: ")
    datasets = {i:[fs.feature_selection(result_path + date + "/" + validation + "/", i, j, datasets, labels, feature_size, classifiers, feature_counter, debug['feature_selection'], feature_selection_parameters, drug_name) for j in range(0, iterations)] for i in feature_selection}

    print("PERFORMING MODEL TRAINING: ")
    models = {j: {classifier: [classification.model_train(result_path + date + "/" + validation + "/" + j + "/", datasets[j][i]['x_train'], datasets[j][i]['y_train'], classifier, debug['classification'], i) for i in range(0, iterations)] for classifier in classifiers} for j in feature_selection}
    print("FINISHED TRAINING MODELS")

    print("GATHERING RESULTS")
    results = {j: {classifier: [val.validate_model(models[j][classifier][i], datasets[j][i]['x_test'], datasets[j][i]['y_test'], 0.50, validation) for i in range(0, iterations)] for classifier in classifiers} for j in feature_selection}
    print("FINISHED GATHERING")
    val.save_results(result_path + date, validation, feature_selection, classifiers, iterations, results, models, datasets, labels, feature_selection_parameters, drug_name)

    if save_fc:
        for i in feature_selection:
            for classifier in classifiers:
                save_feature_counter(result_path, i, classifier, date, validation, feature_counter, feature_size, debug['feature_counter'])
        print("SAVED FEATURE COUNTER")
    print("PIPELINE ENDS")
    return 0

if __name__ == "__main__":
    main()
