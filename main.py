import parser
import data_preprocess as dp
import feature_selection as feat
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
    hyper_opt = parameters['hyper_opt']
    
    feature_selection = parameters['feature_selection'].split(',')

    feature_size = int(parameters['feature_size'])

    classifiers = parameters['classifiers'].split(',')
    train_set_split = float(parameters['train_test_split'])
    train_test_seed = 200

    validation = parameters['validation']
    iterations = int(parameters['validation_iterations'])
    normalization = parameters['normalization']

    fs_keys = ['project','dataset_path', 'dge_path', 'swapped_label', 'drug_feature_path', 'swapped_path']
    feature_selection_parameters = {key: parameters[key] for key in fs_keys}

    save_fc = int(parameters['feature_counter'])

    debug = int(parameters['debug'])

    debug = debug_mode(debug)

    drug_name = parameters['drug_name']

    result_path = result_path + project + "/" + run_name + "/" + drug_name +  "/"
    
    gpu = int(parameters['gpu'])
    
    cluster = None
    client = None
    
    if gpu == 1:
        import gpu.data_preprocess as dp
        import gpu.feature_selection as feat
        import gpu.validation as val
        import gpu.classification as classification
        from dask_cuda import LocalCUDACluster
        from dask.distributed import Client
        cluster = LocalCUDACluster()
        client = Client(cluster)
        feature_selection_parameters['cluster'] = client
 
    else:
        import data_preprocess as dp
        import feature_selection as feat
        import validation as val
        import classification as classification
        
    print("PIPELINE STARTS")

    dataset, dataset_samples = dp.load_dataset(dataset_path, project, normalization)

    print("FINISHED LOADING DATASET")

    labels = dp.load_labels(dataset_path, project, drug_name)

    print("FINISHED LOADING LABELS")

    dataset, labels, samples = dp.sample_match(dataset, labels, dataset_samples)
    
    if parameters['simulation_size'] != 'none':
        simulation_size = int(parameters['simulation_size'])
        dataset, labels, samples = dp.simulate_data(dataset, labels, simulation_size)

        
    print("DATSET BEFORE SIZE:", dataset.shape)
    print("LABELS BEFORE SIZE:", labels.shape)
    
    if int(parameters['balance']):
        dataset, labels, samples = dp.balance_dataset(dataset, labels)
        
    print("DATSET AFTER SIZE:", dataset.shape)
    print("LABELS AFTER SIZE:", labels.shape)
    
    feature_counter = feat.build_feature_counter(dataset)

    print("SPLITTING DATASET BASED ON VALIDATION STYLE: " + validation)

    datasets, iterations = val.split_dataset(validation, dataset, labels, train_set_split, iterations)

    if validation == "cv_and_test":

    # CV HAPPENS FIRST
        for fs in feature_selection:
            for classifier in classifiers:
                for iteration in range(0, iterations):
                    dp.make_result_dir(result_path + date + "/" + validation + "/" + fs + "/" + classifier + "/" + str(iteration) + "/")

        print("CV - PERFORMING FEATURE SELECTION: ")
        datasets_cv = {fs:[feat.feature_selection(result_path + date + "/" + validation + "/", fs, iteration, datasets, labels, feature_size, classifiers, feature_counter, debug['feature_selection'], feature_selection_parameters, drug_name) for iteration in range(0, iterations)] for fs in feature_selection}

        print("CV - PERFORMING MODEL TRAINING: ")
        best_parameters = {}
        models = {fs: {classifier: [classification.model_train(result_path + date + "/" + validation + "/" + fs + "/", datasets_cv[fs][iteration]['x_train'], datasets_cv[fs][iteration]['y_train'], classifier, debug['classification'], iteration, hyper_opt, best_parameters) for iteration in range(0, iterations)] for classifier in classifiers} if fs != "random" else {classifier: ["no random cv" for iteration in range(0, iterations)] for classifier in classifiers}  for fs in feature_selection}
        print("CV - FINISHED TRAINING MODELS")

        print("CV - GATHERING RESULTS")
        cv_results = {fs: {classifier: [val.validate_model(models[fs][classifier][iteration][0], datasets_cv[fs][iteration]['x_test'], datasets_cv[fs][iteration]['y_test'], 0.50, "cv") for iteration in range(0, iterations)] for classifier in classifiers} if fs != "random" else {classifier: ["no random cv" for iteration in range(0, iterations)] for classifier in classifiers} for fs in feature_selection}
        #print("CV - FINISHED GATHERING")
        #val.save_results(result_path + date, "cv", feature_selection, classifiers, iterations, results, models, datasets, labels, feature_selection_parameters, drug_name)

        if save_fc:
            for fs in feature_selection:
                for classifier in classifiers:
                    save_feature_counter(result_path, fs, classifier, date, validation, feature_counter, feature_size, debug['feature_counter'])
        print("CV - SAVED FEATURE COUNTER")

     # INDEPENDENT TEST SET
        for fs in feature_selection:
            for classifier in classifiers:
#                for k in range(0, iterations):
                dp.make_result_dir(result_path + date + "/" + validation + "/" + fs + "/" + classifier + "/hold_out/")

        print("HOLD-OUT - PERFORMING FEATURE SELECTION: ")
      #  print(datasets)
        datasets = {fs:feat.feature_selection(result_path + date + "/" + validation + "/", fs, "hold_out", datasets['hold_out'], labels, feature_size, classifiers, feature_counter, debug['feature_selection'], feature_selection_parameters, drug_name) for fs in feature_selection}

        print("HOLD-OUT - PERFORMING MODEL TRAINING: ")

        models = {j: {classifier: [classification.model_train(result_path + date + "/" + validation + "/" + j + "/", datasets[j]['x_train'], datasets[j]['y_train'], classifier, debug['classification'], "hold_out", "best", models[j][classifier][i]) for i in range(0, iterations)] for classifier in classifiers} if j != "random" else {classifier: [classification.model_train(result_path + date + "/" + validation + "/" + j + "/", datasets[j]['x_train'], datasets[j]['y_train'], classifier, debug['classification'], "hold_out", "none", models[j][classifier][i]) for i in range(0, iterations)] for classifier in classifiers} for j in feature_selection}
        holdout_results = {j: {classifier: [val.validate_model(models[j][classifier][i][0], datasets[j]['x_test'], datasets[j]['y_test'], 0.50, validation) for i in range(0, iterations)] for classifier in classifiers} for j in feature_selection}

        models, holdout_results = val.pick_top_performer(models, cv_results, holdout_results, classifiers, feature_selection, iterations)

        print("FINISHED TRAINING MODELS")
        print("HOLD-OUT - FINISHED GATHERING RESULTS")
        results = {"cv": cv_results, "holdout":holdout_results}
        val.save_results(result_path + date, validation, feature_selection, classifiers, iterations, results, models, datasets, labels, feature_selection_parameters, drug_name)

        if save_fc:
            for fs in feature_selection:
                for classifier in classifiers:
                    save_feature_counter(result_path, fs, classifier, date, validation, feature_counter, feature_size, debug['feature_counter'])
        print("HOLD-OUT - SAVED FEATURE COUNTER")

    else:
        for fs in feature_selection:
            for classifier in classifiers:
                for iteration in range(0, iterations):
                    dp.make_result_dir(result_path + date + "/" + validation + "/" + fs + "/" + classifier + "/" + str(iteration) + "/")

        print("PERFORMING FEATURE SELECTION: ")
        datasets = {fs:[feat.feature_selection(result_path + date + "/" + validation + "/", fs, iteration, datasets, labels, feature_size, classifiers, feature_counter, debug['feature_selection'], feature_selection_parameters, drug_name) for iteration in range(0, iterations)] for fs in feature_selection}

        print("PERFORMING MODEL TRAINING: ")
        best_parameters = {}
        models, best_parameters = {fs: {classifier: [classification.model_train(result_path + date + "/" + validation + "/" + fs + "/", datasets[fs][iteration]['x_train'], datasets[fs][iteration]['y_train'], classifier, debug['classification'], iteration, hyper_opt, best_parameters) for iteration in range(0, iterations)] for classifier in classifiers} for fs in feature_selection}
        print("FINISHED TRAINING MODELS")

        print("GATHERING RESULTS")
        results = {j: {classifier: [val.validate_model(models[fs][classifier][iteration][0], datasets[fs][iteration]['x_test'], datasets[fs][iterationo]['y_test'], 0.50, validation) for i in range(0, iterations)] for classifier in classifiers} for fs in feature_selection}
        print("FINISHED GATHERING")
        val.save_results(result_path + date, validation, feature_selection, classifiers, iterations, results, models, datasets, labels, feature_selection_parameters, drug_name)

        if save_fc:
            for fs in feature_selection:
                for classifier in classifiers:
                    save_feature_counter(result_path, fs, classifier, date, validation, feature_counter, feature_size, debug['feature_counter'])
        print("SAVED FEATURE COUNTER")
    print("PIPELINE ENDS")
    return 0

if __name__ == "__main__":
    main()
