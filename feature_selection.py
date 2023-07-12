import matplotlib.pyplot as plt
import data_preprocess as dp
import pandas as pd
import numpy as np
import shap
import sklearn
import xgboost
from xgboost import plot_importance
import sys
import time

# Feature selection wrapper, chooses the correct feature selection technique based on the configuration file parameters
def feature_selection(path, fs, iteration, input, labels, feature_size, classifiers, feature_counter, debug_mode, project_info, drug_name):

    if iteration == "hold_out":

        #total_iterations = len(input["y_train"])

        # DEBUG MODE
        if debug_mode:

            # SAVES THE INPUT OF THE FEATURE SELECTION TECHNIQUE AND THE FEATURE COUNTER
            debug_path = path + fs + "/debug/" + iteration + "/"
            dp.make_result_dir(debug_path)
            input["x_train"].to_csv(debug_path + "input_dataset.tsv", sep="\t")
            input["y_train"].to_csv(debug_path + "labels.tsv", sep="\t")
            debug_feature_counter = pd.DataFrame(feature_counter.items(), columns=["FEATURE", "FREQUENCY"])
            debug_feature_counter = debug_feature_counter.sort_values(by=['FREQUENCY'], ascending=False)
            debug_feature_counter.to_csv(debug_path + "input_feature_counter.tsv", index = False, sep="\t")

            # SHAPLEY VALUE FEATURE SELECTION
        if fs == 'shap':
            print("PERFORMING SHAP: ")
            start = time.time()
            dataset = shapley(path + fs + "/" + classifiers[0] + "/" + iteration, input["x_train"], input["y_train"], feature_size, 1)
            end = time.time()
            print("SHAP RUN TIME: ", end - start)

            # PRINCIPAL COMPONENT ANALYSIS
        elif fs == 'pca':
            print("PERFORMING PCA: ")
            dataset, datatest = principal_component_analysis(input["x_train"], input["x_test"], feature_size)

            # DIFFERENTIAL GENE EXPRESSION ANALYSIS
        elif fs == 'dge':
            print("PERFORMING DGE: ")
            dataset = dge(path + fs + "/" + classifiers[0] + "/" + iteration + "/", input["x_train"].T, input["y_train"], drug_name, project_info)

        elif fs == 'chi2':
            dataset = chi_square(input["x_train"], input["y_train"], feature_size)

        elif fs == 'rae':
            dataset = rare_allele_enrichment(path, input["x_train"], input["y_train"], 0.05, feature_size)
            # FEATURE SWAPPING EXPERIMENT
        elif fs == 'swap':
            print("PERFORMING FEATURE SWAPPING: ")
            dataset = from_feature_list(path, input["x_train"].T, input["y_train"], iteration, project_info)

            # SELECT RANDOM FEATURES
        elif fs == 'random':
            print("SELECTING RANDOM FEATURES: ")
            dataset = random_selected_features(input["x_train"], feature_size)
            # USING SHAP, PCA, AND DGE TO SELECT FEATURES AND THEN USING ALL THE FEATURES FOUND AS THE NEW DATASET
        elif fs == 'all':
            print("SELECTING AND COMBINING FEATURES USING ALL FEATURE REDUCTION TOOLS: ")

            shap_dataset = shapley(path + fs + "/" + classifiers[0] + "/" + iteration, input["x_train"], input["y_train"], feature_size, 1)
            dge_dataset = dge(path, input["x_train"].T, input["y_train"], drug_name, project_info)
            shap_features = set(shap_dataset.columns)
            dge_features = set(dge_dataset.columns)
            all_features = shap_features | dge_features
            dataset = input["x_train"].loc[input["x_train"].index.isin(all_features)].T
           # print(dataset)

        elif fs == 'none':
            print("NO FEATURE SELECTION: ")
            dataset = input["x_train"].T
        else:
            sys.exit("ERROR: Unrecognized Feature Selection technique in configuration file")

            ## PCA does not output the features the same way as the other FS techniques
        if fs == 'pca':
            dict = {"x_train": dataset.T, "x_test": datatest.T,  "y_train": input["y_train"], "y_test": input["y_test"]}
        else:
            # Filter through the dataset to save only the data rows that correspond to the features selecteed
            print("WHATS UP")
            print(dataset.shape)
            print(input["y_train"].shape)
            features = dataset.columns
            dict = {"x_train": input["x_train"].loc[input["x_train"].index.isin(features)], "x_test": input["x_test"].iloc[input["x_test"].index.isin(features)],  "y_train": input["y_train"], "y_test": input["y_test"]}

            # adds to the counter for each feature selected
            add_to_feature_counter(features, feature_counter)

            # Saves the list of features/genes as a tsv file
            for classifier in classifiers:
                with open(path + fs + "/" + classifier + "/" + iteration + "/genes_selected.tsv", 'w') as file:
                    for row in features:
                        s = "".join(map(str, row))
                        file.write(s+'\n')

        # DEBUG MODE
        if debug_mode:
            # SAVES THE OUTPUT OF THE FEATURE SELECTION TECHNIQUE AND THE MODIFIED FEATURE COUNTER
            debug_feature_counter = pd.DataFrame(feature_counter.items(), columns=["FEATURE", "FREQUENCY"])
            debug_feature_counter = debug_feature_counter.sort_values(by=['FREQUENCY'], ascending=False)
            debug_feature_counter.to_csv(debug_path + "output_feature_counter.tsv", index = False, sep="\t")
            dataset.to_csv(debug_path + "/output_dataset.tsv", sep='\t')

    else:

        total_iterations = len(input["y_train"])

        # DEBUG MODE
        if debug_mode:

            # SAVES THE INPUT OF THE FEATURE SELECTION TECHNIQUE AND THE FEATURE COUNTER
            debug_path = path + fs + "/debug/" + str(iteration) + "/"
            dp.make_result_dir(debug_path)
            input["x_train"][iteration].to_csv(debug_path + "input_dataset.tsv", sep="\t")
            input["y_train"][iteration].to_csv(debug_path + "labels.tsv", sep="\t")
            debug_feature_counter = pd.DataFrame(feature_counter.items(), columns=["FEATURE", "FREQUENCY"])
            debug_feature_counter = debug_feature_counter.sort_values(by=['FREQUENCY'], ascending=False)
            debug_feature_counter.to_csv(debug_path + "input_feature_counter.tsv", index = False, sep="\t")

            # SHAPLEY VALUE FEATURE SELECTION
        if fs == 'shap':
            print("PERFORMING SHAP: " + str(iteration) + "/" + str(total_iterations))
            start = time.time()
            dataset = shapley(path + fs + "/" + classifiers[0] + "/" + str(iteration), input["x_train"][iteration], input["y_train"][iteration], feature_size, 1)
            end = time.time()
            print("SHAP RUN TIME: ", end - start)

            # PRINCIPAL COMPONENT ANALYSIS
        elif fs == 'pca':
            print("PERFORMING PCA: " + str(iteration) + "/" + str(total_iterations))
            dataset, datatest = principal_component_analysis(input["x_train"][iteration], input["x_test"][iteration], feature_size)

            # DIFFERENTIAL GENE EXPRESSION ANALYSIS
        elif fs == 'dge':
            print("PERFORMING DGE: " + str(iteration) + "/" + str(total_iterations))
            dataset = dge(path + fs + "/" + classifiers[0] + "/" + str(iteration) + "/", input["x_train"][iteration].T, input["y_train"][iteration], drug_name, project_info)

            # FEATURE SWAPPING EXPERIMENT
        elif fs == 'swap':
            print("PERFORMING FEATURE SWAPPING: " + str(iteration) + "/" + str(total_iterations))
            dataset = from_feature_list(path, input["x_train"][iteration].T, input["y_train"][iteration], iteration, project_info)

        elif fs == 'chi2':
            dataset = chi_square(input["x_train"][iteration].T, input["y_train"][iteration], feature_size)
        elif fs == "rae":
             dataset = rare_allele_enrichment(path, input["x_train"][iteration], input["y_train"][iteration], 0.05, feature_size)
            # SELECT RANDOM FEATURES
        elif fs == 'random':
            print("SELECTING RANDOM FEATURES: " + str(iteration) + "/" + str(total_iterations))
            dataset = random_selected_features(input["x_train"][iteration], feature_size)
            # USING SHAP, PCA, AND DGE TO SELECT FEATURES AND THEN USING ALL THE FEATURES FOUND AS THE NEW DATASET
        elif fs == 'all':
            print("SELECTING AND COMBINING FEATURES USING ALL FEATURE REDUCTION TOOLS: " + str(iteration) + "/" + str(total_iterations))

            shap_dataset = shapley(path + fs + "/" + classifiers[0] + "/" + str(iteration), input["x_train"][iteration], input["y_train"][iteration], feature_size, 1)
            dge_dataset = dge(path, input["x_train"][iteration].T, input["y_train"][iteration], drug_name, project_info)
            shap_features = set(shap_dataset.columns)
            dge_features = set(dge_dataset.columns)
            all_features = shap_features | dge_features
            dataset = input["x_train"][iteration].loc[input["x_train"][iteration].index.isin(all_features)].T
         #   print(dataset)

        elif fs == 'none':
            print("NO FEATURE SELECTION: " + str(iteration) + "/" + str(total_iterations))
            dataset = input["x_train"][iteration].T
        else:
            sys.exit("ERROR: Unrecognized Feature Selection technique in configuration file")

            ## PCA does not output the features the same way as the other FS techniques
        if fs == 'pca':
            dict = {"x_train": dataset.T, "x_test": datatest.T,  "y_train": input["y_train"][iteration], "y_test": input["y_test"][iteration]}
        else:
            # Filter through the dataset to save only the data rows that correspond to the features selecteed
            features = dataset.columns
            print(dataset.shape)
            print(input["y_train"][iteration].shape)
            print(input["x_train"][iteration].loc[input["x_train"][iteration].index.isin(features)].shape)
            print(input["x_train"][iteration].loc[input["x_train"][iteration].index.isin(features)])
            dict = {"x_train": input["x_train"][iteration].loc[input["x_train"][iteration].index.isin(features)], "x_test": input["x_test"][iteration].iloc[input["x_test"][iteration].index.isin(features)],  "y_train": input["y_train"][iteration], "y_test": input["y_test"][iteration]}

            # adds to the counter for each feature selected
            add_to_feature_counter(features, feature_counter)

            # Saves the list of features/genes as a tsv file
            for classifier in classifiers:
                with open(path + fs + "/" + classifier + "/" + str(iteration) + "/genes_selected.tsv", 'w') as file:
                    for row in features:
                        s = "".join(map(str, row))
                        file.write(s+'\n')

        # DEBUG MODE
        if debug_mode:
            # SAVES THE OUTPUT OF THE FEATURE SELECTION TECHNIQUE AND THE MODIFIED FEATURE COUNTER
            debug_feature_counter = pd.DataFrame(feature_counter.items(), columns=["FEATURE", "FREQUENCY"])
            debug_feature_counter = debug_feature_counter.sort_values(by=['FREQUENCY'], ascending=False)
            debug_feature_counter.to_csv(debug_path + "output_feature_counter.tsv", index = False, sep="\t")
            dataset.to_csv(debug_path + "/output_dataset.tsv", sep='\t')

    return dict


# Feature Selection: "random"
# Selects "feature_size" random features from the dataset
def random_selected_features(dataset, feature_size):
    dataset = dataset.sample(feature_size)
    dataset = dataset.T
    return dataset

# Feature Selection: "swap"
# Loads the features/genes that were selected for the drug that the user wants to "swap"
def from_feature_list(path, dataset, labels, iteration, project_info):
    feature_set = pd.read_csv(project_info['swapped_path'] + str(iteration) + '/genes_selected.tsv', names=[project_info['swapped_label']])
    filtered = dataset[feature_set[project_info['swapped_label']].values]
    return filtered

# possibly combine this with from_feature_list above
# Feature Selection: "dge"
# Loads the features/genes that were selected by the DGE analysis
def dge(path, dataset, labels, drug_name, project_info):

    # Generate DGE label file used in for the limma R script
    dge_labels_file = path + drug_name + '_dge_input.txt'
    dge_labels = labels.copy()
    dge_labels = dge_labels.reset_index()
    dge_labels["SID"] = [s.replace('-','X') for s in dge_labels["SID"]]
    dge_labels['SID'] = 'X' + dge_labels['SID'].astype(str)
    dge_labels = dge_labels.rename(columns = {'SID':'Sample'})
    dge_labels = dge_labels.rename(columns = {'GROUP':'high'})
    dge_labels['low'] = np.logical_xor(dge_labels['high'],1).astype(int)
    dge_labels.to_csv(dge_labels_file, index=False, sep="\t")
    #print(dge_labels)

    import sys
    import subprocess

    dge_script  = "./beataml_deg_commandline.R"
    workdir   = "--dir=" + project_info['dataset_path']
    file = "--file=" + dge_labels_file
    name = "--name=" + path + drug_name
    sys.stdout.flush()
    jobargz = []
    jobargz.append(file)
    jobargz.append(name)
    jobargz.append(workdir)
    runlaunch = subprocess.Popen([project_info['dge_path'] + dge_script] + jobargz)
    runlaunch.wait()

    limma_script  = "limma.py"
    dataset_path = "--dataset=" + project_info['dataset_path']
    dname = "--drug=" + drug_name
    result_path = "--dir=" + path
    sys.stdout.flush()
    jobargz = []
    jobargz.append(dataset_path)
    jobargz.append(result_path)
    jobargz.append(dname)
    #jobargz.append(workdir)
    runlaunch = subprocess.Popen(["python", project_info['dge_path'] + limma_script] + jobargz)
    runlaunch.wait()

    feature_set = pd.read_csv(path + drug_name + '_genes_selected.tsv', names=[drug_name])
    #feature_set = pd.read_csv(project_info['dge_path'] + drug_name + '_genes_selected.tsv', names=[drug_name])
    filtered = dataset[feature_set[drug_name].values]
    return filtered

def chi_square(dataset, labels, feature_size):
    from sklearn.feature_selection import chi2
    print(dataset)
    print(labels)
    chi_scores, p_values = chi2(dataset, labels)
    p_values = pd.Series(chi_scores[1],index = dataset.columns)
    p_values.sort_values(ascending = False , inplace = True)
    print(p_values)
   # p_values.plot.bar()
   # plt.show()

   # sys.exit("Kill")
    return dataset

# Feature Selection: "pca"
# Performs principal component analysis on the dataset, from the scikit-learn package
def principal_component_analysis(dataset, datatest, feature_size):
    import time
    start = time.time()
    pca = sklearn.decomposition.PCA(n_components=30)
      # This is the training data
    X_pca = pca.fit_transform(dataset.T)
    end = time.time()
    print("Finished PCA: " + str(end - start))
    # This is the test data
    test_pca = pca.transform(datatest.T)
    end_test =time.time()
    print("Finished test PCA: " + str(end_test - end))
    # Selects the "feature_size" components from the PCA results and outputs that as the new dataset
    X_selected = X_pca[:,:feature_size]
    test_selected = test_pca[:,:feature_size]
    return X_selected, test_selected

def rare_allele_enrichment(path, dataset, labels, pvalue_threshold, feature_size):

    control_labels = labels[labels == 0]
    disease_labels = labels[labels == 1]
    control_samples = control_labels.index.to_numpy()
    disease_samples = disease_labels.index.to_numpy()

    control_dataset = dataset[[c for c in dataset.columns if c in control_samples]]
    disease_dataset = dataset[[c for c in dataset.columns if c in disease_samples]]

    print("Counting variant occurance for each feature: ")
    tcount0 = time.time()
    control_size = control_dataset.apply(pd.value_counts, axis = 1)
    disease_size = disease_dataset.apply(pd.value_counts, axis = 1)
   # control_size = control_dataset.T.value_counts()
  #  print(control_size[0])
    tcount1 = time.time()
    print("Count took: " + str(tcount1 - tcount0))

  #  sys.exit("EXIT")
    
    control_size = control_size.fillna(0)
    disease_size = disease_size.fillna(0)

    tables = []

    tcont0 = time.time()
    for i, feature in enumerate(control_dataset.index):
    #    if i % 500 == 0:
   #         print("Building contingency table for feature: " + str(i) + "/" + str(len(control_dataset.index)), flush = True)
        variant_control_yes = control_size.loc[feature].iloc[0]
        variant_control_no = control_size.loc[feature].iloc[1] + control_size.loc[feature].iloc[2]
        variant_disease_yes = disease_size.loc[feature].iloc[1] + disease_size.loc[feature].iloc[2]
        variant_disease_no = disease_size.loc[feature].iloc[0]

        control = [variant_control_no, variant_control_yes]
        disease = [variant_disease_yes, variant_disease_no]

        contigency_table = np.array([control, disease])
        tables.append(contigency_table)
    tcont1 = time.time()
    print("Building the contigency tables took: " + str(tcont1 - tcont0))
    odd_ratios = []
    pvalues = []
    tfish0 = time.time()
    import scipy
    for i, table in enumerate(tables):
  #      if i % 300 == 0:
 #           print("Performing Fisher's exact test for feature: " + str(i) + "/" + str(len(tables)), flush = True)
        oddsratio, pvalue = scipy.stats.fisher_exact(table)
        odd_ratios.append(oddsratio)
        pvalues.append(pvalue)
    tfish1 = time.time()
    print("Fisher's exact test took: " + str(tfish1 - tfish0))
    temp = dataset.copy()
    temp['pvalues'] = pvalues
    temp = temp[temp['pvalues'] > pvalue_threshold]
    selected_features = temp.nsmallest(feature_size, 'pvalues')
    pvalue_threshold = selected_features['pvalues'].iloc[0]
    selected_features = selected_features.drop('pvalues', axis = 1)
    temp = temp.drop('pvalues', axis = 1)
    print(temp)
    selected_features = selected_features.T
    print(selected_features)
    return selected_features


# Feature Selection: "shap"
# Performs the shapley value feature selection technique discussed in the paper
def shapley(path, dataset, labels, feature_size, plot):
    dataset = dataset.T
    # Set xgboost model to run the shap value calculations using default parameters
    # This can be changed to other ensemble models that the shap package supports (Random Forest, etc)
    model = xgboost.XGBClassifier(eval_metric='logloss', verbosity = 3)
    import time 
    start = time.time()
    model.fit(dataset, labels)
    end = time.time()
    print("Xgboost training finished " + str(end - start))
    # initializes the shap JavaScript visualization
   # shap.initjs()
    # Calculates the shap value contributions for
    shap_values = shap.TreeExplainer(model).shap_values(dataset)
    # Removes direction to the shap value marginal contribution by taking the absolute value
    # We only care about magnitude to select features
    distribution = np.absolute(shap_values)
    # Takes the mean of the shap value contribution scores acrosss all samples
    # This provides a single shap value contribution for each feature
    distribution = distribution.mean(axis=0)
    # If selected, plot the SHAP feature importance summary plot
    if(plot):
        plot_shap(path, shap_values, dataset)
        plot_model_importance(path, feature_size, model)
    # Selects the top "feature_size" genes with the largest absolute mean shap value score
    index = np.argpartition(distribution, -(feature_size))[-(feature_size):]
    slice = dataset.iloc[:,index]
    return slice

# Function used to generate the SHAP Feature importance summary plots
def plot_shap(path, shap_values, dataset):
        shap.summary_plot(shap_values, features=dataset, feature_names=dataset.columns, show=False)
        plt.title('Summed Shap Values Plot')
        plt.xlabel('Shap Values')
        plt.ylabel('Feature')
        figure = plt.gcf()
        figure.set_size_inches(15, 10)
        plt.savefig(path + "/shap_feature_importance.png", dpi=100)
        plt.clf()
        return
  

def plot_model_importance(path, feature_size, model):
    xgboost.plot_importance(model, max_num_features=feature_size)
    figure = plt.gcf()
    figure.set_size_inches(15, 10)
    plt.savefig(path + "/model_feature_importance.png")
    plt.clf()
    return

# Creates the feature counter dictionary for all features in the dataset
def build_feature_counter(dataset):
    dict = {feature:0 for feature in dataset.index}
    return dict

# Adds 1 to the counter if the feature is in the feature list
def add_to_feature_counter(features, counter):
    for feature in features:
        counter[feature] = counter[feature] + 1
    return

def hierarchical_clustering_heatmap(data, iteration):
    sns.clustermap(data)
    plt.savefig(path + "/hierarchical_clustering.png", dpi=100)
    plt.clf()
    return
