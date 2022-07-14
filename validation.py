import numpy as np
import pandas as pd
import classification as cl
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from collections import Counter
import data_preprocess as dp
import sys

# Handler that selects the correct validation and dataset splitting mode that corresponds to the configuration parameters
def split_dataset(mode, dataset, labels, split, iterations):
    # Cross-Validation
    if mode == 'cv':
        dataset, iterations = cross_validation(dataset, labels, iterations)
    # Bootstrapping
    elif mode == 'bootstrap':
        dataset = bootstrapping(dataset, labels, split, iterations)
    # Leave-One-Out
    elif mode == 'loo':
        dataset, iterations = leave_one_out(dataset, labels, iterations)
    elif mode == 'cv_and_test':
        dataset, iterations = cv_and_test(dataset, labels, split, iterations)
    else:
        sys.exit("ERROR: Unrecognized validation technique in configuration file")
    return dataset, iterations

# Validation mode: "cv_and_test"
# Divides the dataset by CV "iterations"-folds
# Training set -> Rest of Dataset, Test set -> Fold X
def cv_and_test(X, y, split, iterations):
    train_data, test_data, train_labels, test_labels = train_test_split(X.T, y.set_index('SID')['GROUP'], test_size=split, shuffle=True)
    X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []
    # Sets up scikit-learn iterator for splitting the dataset
    kf = StratifiedKFold(n_splits=iterations)
    for train_index, test_index in kf.split(train_data, train_labels):
        X_train, X_test = train_data.iloc[train_index], train_data.iloc[test_index]
        y_train, y_test = train_labels.iloc[train_index], train_labels.iloc[test_index]
        X_train = X_train.T
        X_test = X_test.T
        X_train_list.append(X_train)
        X_test_list.append(X_test)
        y_train_list.append(y_train)
        y_test_list.append(y_test)
    iterations = kf.get_n_splits(train_data, train_labels)
    hold_out = {'x_train':train_data.T,'x_test':test_data.T,'y_train':train_labels,'y_test':test_labels}
    dict = {'x_train':X_train_list,'x_test':X_test_list,'y_train':y_train_list,'y_test':y_test_list, 'hold_out':hold_out}
    return dict, iterations

# Validation mode: "cv"
# Divides the dataset by CV "iterations"-folds
# Training set -> Rest of Dataset, Test set -> Fold X
def cross_validation(X, y, iterations):
    X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []
    # Sets up scikit-learn iterator for splitting the dataset
    kf = StratifiedKFold(n_splits=iterations)
    X = X.T
    y = y.set_index('SID')['GROUP']
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        X_train = X_train.T
        X_test = X_test.T
        X_train_list.append(X_train)
        X_test_list.append(X_test)
        y_train_list.append(y_train)
        y_test_list.append(y_test)
    iterations = kf.get_n_splits(X, y)
    dict = {'x_train':X_train_list,'x_test':X_test_list,'y_train':y_train_list,'y_test':y_test_list}
    return dict, iterations

# Validation mode: "loo"
# Divides the dataset by "Leaving one out", or using one sample as the test set
# Training set -> Rest of dataset, Test set -> Sample X
def leave_one_out(X, y, iterations):
    X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []
    loo = LeaveOneOut()
    X = X.T
    y = y.set_index('SID')['GROUP']
    for train_index, test_index in loo.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        X_train_list.append(X_train.T)
        X_test_list.append(X_test.T)
        y_train_list.append(y_train)
        y_test_list.append(y_test)
    iterations = loo.get_n_splits(X)
    dict = {'x_train':X_train_list,'x_test':X_test_list,'y_train':y_train_list,'y_test':y_test_list}
    return dict, iterations

# Validation mode: "bootstrap"
# Divides the dataset by splitting it into a test and train dataset, using the "split" percentage to:
# determine the size of each set. Shuffles the order of the dataset so that the samples selected in each set are different
# Training set -> Rest of dataset, Test set -> split percentage
def bootstrapping(dataset, labels, split, iterations):
    X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []
    dataset = dataset.T
    for i in range(0, iterations):
        X_train, X_test, y_train, y_test = train_test_split(dataset, labels.set_index('SID')['GROUP'], test_size=split, shuffle=True)
        X_train_list.append(X_train.T)
        X_test_list.append(X_test.T)
        y_train_list.append(y_train)
        y_test_list.append(y_test)
    dict = {'x_train':X_train_list,'x_test':X_test_list,'y_train':y_train_list,'y_test':y_test_list}
    return dict

# Assigns a binary representation to the probability predictions made by the classifiers
# if higher than threshold == 1
# if lower or equal than threshold == 0
def assign_class(pred, threshold):
    temp = []
    # Iterates through each test observation
    for prediction in pred:
        # If the observation probability is higher than the threshold, assign label 1 to the observation, otherwise assign 0
        if prediction > threshold:
            temp.append(1)
        else:
            temp.append(0)
    return temp

# Runs inference on the trained classifier using the test set by:
# Getting prediction probability scores for the test n_samples
# Assigning a label based on the prediction probability
# Creating  confusion matrix, roc curve plot, and accuracy score using the predictions
def validate_model(model, x, y, threshold, mode):
    # Transform the dataset and label for scikit-learn formatting
    x, y = cl.prepare_dataset(x, y)
    # Assigns probability for each test case in the form of (Probability for class 0, probability for class 1)
    prediction_proba = model.predict_proba(x)
    # Grabs the probability for class 1
    prediction_proba = prediction_proba[:, 1]
    # Assigns class label to each test case based on a threshold
    prediction = assign_class(prediction_proba, threshold)

    if mode == 'loo' or mode == 'cv' or mode == 'cv_and_test':
        dict = {'sample':y.index,'true_label':y.values, 'pred':prediction, 'pred_prob': prediction_proba}
    else:
        # Calculates the False Positive Rate (FPR) and True Positive Rate (TPR) of the test case probability predictions
        # Generates a List of thresholds that will be used for the ROC calculations
        thresholds = make_thresholds(10000)
        # Calculates the False Positive Rate (FPR) and True Positive Rate (TPR) of the test case probability predictions using the threshold list above
        tpr, fpr = my_roc_curve(y, prediction_proba, thresholds)
        #fpr, tpr, thresholds = sklearn.metrics.roc_curve(y, prediction_proba) # What's used to plot the ROC Curves
        # Calculates the Area Under the Curve score
        roc_score = roc_auc_score(y, prediction_proba)
        # Calculates accuracy based on ground truth (y_test) and the assigned labels above
        accuracy = accuracy_score(y, prediction)
        # Creates a confusion matrix
        tn, tp, fn, fp = my_confusion_matrix(y, prediction)
        confusionmatrix = [[tp,fp],[fn,tn]]
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        dict = {'fpr':fpr, 'tpr':tpr, 'roc':roc_score, 'acc':accuracy, 'sens': sensitivity, 'spec': specificity, 'pred':prediction, 'pred_prob': prediction_proba}
    return dict

# Helper function for ROC Curve implementation from scratch (not using the scikit-learn implementation)
# Creates a list of thresholds that will be used for each point in the ROC Curve
def make_thresholds(n):
  thresholds = []
  # Creates threshold list from 0/n, 1/n, 2/n, .... to n/n
  for i in range(0, n + 1):
    thresholds.append(i/n)
  return thresholds

def my_confusion_matrix(true, pred):
  tn = 0
  tp = 0
  fn = 0
  fp = 0

  # Iterates through all test observations
  for i in range(len(pred)):
      # the predicted label equals the ground truth label and they are both 0, adds to true negative
    if pred[i] == 0 and true[i] == 0:
      tn = tn + 1
      # the predicted label equals the ground truth label and they are both 1, adds to true positive
    elif pred[i] == 1 and true[i] == 1:
      tp = tp + 1
      # the predicted label does not equals the ground truth label and they are 0 and 1 respectively, adds to false negative
    elif pred[i] == 0 and true[i] == 1:
      fn = fn + 1
      # the predicted label does not equals the ground truth label and they are 1 and 0 respectively, adds to false positive
    elif pred[i] == 1 and true[i] == 0:
      fp = fp + 1
    else:
      print("Invalid combination")
  return tn, tp, fn, fp

# Function that calculates and generates an ROC curve from scratch (not scikit-learn implementation)
def my_roc_curve(true, pred, thresholds):
  tpr = []
  fpr = []
  # Iterates through list of thresholds
  for threshold in thresholds:
    # Assigns labels to each test case
    predictions = assign_class(pred, threshold)
    # Finds the true positive, true negative, false negative, and false positive values
    tn, tp, fn, fp = my_confusion_matrix(true, predictions)
    cp = tp + fn
    cn = fp + tn
    # Calculates True Positive Rate and False Positive Rate, then adds to a list. The list will have the calculated TPR and FPR points for each threshold iterated
    tpr.append(tp/cp)
    fpr.append(1 - tn/cn)
  return tpr, fpr

# Plots the ROC curve
def mlpipeline_plot_roc(result_path, x, y, auc, classifier, feature_selection):
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.plot(x, y, color='orange', label= classifier.upper() + ' auc=' + str(round(auc, 3)), alpha = 0.5)
    plt.xlabel('1 - Specificity (False Positive Rate)')
    plt.ylabel('Sensitivity (True Positive Rate)')
    plt.legend()
    plt.title('ROC Model: ' + feature_selection.upper() +  ' Feature Selection')
    plt.savefig(result_path + "roc.png")
    plt.clf()
    return

# Plots the confusion matrix
def mlpipeline_plot_cm(result_path, mode, model, x_test, y_test, iteration):
    class_names = ['Negative Cohort','Positive Cohort']

    # Since results are stored differently depending on validation type there are two versions of the code that plots the matrix
    if mode == 'loo' or mode == 'cv' or mode == "cv_and_test":
        print(x_test)
        print(y_test)
        print(len(x_test))
        print(len(y_test))
        #x_test, y_test = cl.prepare_dataset(x_test, y_test)
        cm = confusion_matrix(x_test, y_test, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=class_names)
        disp = disp.plot(cmap=plt.cm.Blues, colorbar = False)
    else:
        x_test, y_test = cl.prepare_dataset(x_test, y_test)
        if iteration != "hold_out":
            x_test = x_test.apply(pd.to_numeric)
        disp = plot_confusion_matrix(model, x_test, y_test,
                                     display_labels=class_names,
                                     cmap=plt.cm.Blues, colorbar = False)
    if iteration == "hold_out":
        disp.ax_.set_title('Confusion Matrix Iteration: ' +  iteration)
    else:
        disp.ax_.set_title('Confusion Matrix Iteration: ' +  str(iteration))
    plt.savefig(result_path + "cm.png")
    plt.clf()
    return

def pick_top_performer(models, cv_results, holdout_results, classifiers, feature_selection, iterations):
    best_models = {}
    best_results = {}
    top_performer = -1
    for fs in feature_selection:
        models_inside_df = {}
        results_inside_df = {}
        for classifier in classifiers:
            for i in range(0, iterations):
                score = roc_score = roc_auc_score(cv_results[fs][classifier][i]['true_label'], cv_results[fs][classifier][i]['pred_prob'])
                if score > top_performer:
                    model = models[fs][classifier][i]
                    result = holdout_results[fs][classifier][i]
                    top_performer = score
            models_inside_df[classifier] = model
            results_inside_df[classifier] = result
        best_models[fs] = models_inside_df
        best_results[fs] = results_inside_df
    return best_models, best_results

# Saves the ROC Curve data points
def save_roc(result_path, fpr, tpr, thresholds):
    roc = pd.DataFrame({"sensitivity":fpr, "1 - specificity":tpr, "thresholds":thresholds})
    roc.to_csv(result_path + "/roc_data.tsv", sep ='\t', index = False)

# Plots venn diagram for the features between the two drugs tested in the "Swapped" experiment
def plot_venn_diagram(path, project_info, iteration, drug_name):
    real_set = pd.read_csv(project_info['drug_feature_path'] + str(iteration) + '/genes_selected.tsv', names=[drug_name])
    flipped_set = pd.read_csv(project_info['swapped_path'] + str(iteration) + '/genes_selected.tsv', names=[project_info['swapped_label']])
    # For set notation
    A = set(real_set[drug_name])
    B = set(flipped_set[project_info['swapped_label']])
    # Intersection of A and B
    AB_overlap = A & B
    # places them in order for plotting
    sets = Counter()
    sets['10'] = len(A-AB_overlap)
    sets['01'] = len(B-AB_overlap)
    sets['11'] = len(AB_overlap)
    plt.figure(figsize=(7,7))
    ax = plt.gca()
    v = venn2(subsets=sets, set_labels=[drug_name, project_info['swapped_label']], ax=ax, set_colors=('red','blue'),alpha=0.5)
    plt.title('Overlapping features for Iteration - ' + str(iteration) + " : " + drug_name + ' and ' + project_info['swapped_label'])
    venn_path = path + str(iteration) + "/venn_d/"
    dp.make_result_dir(venn_path)
    with open(venn_path + "/overlapping_genes.tsv", 'w') as file:
        for row in list(AB_overlap):
            s = "".join(map(str, row))
            file.write(s+'\n')
    plt.savefig(venn_path + "venn_diagram.png")
    plt.clf()

def individual_sample_report(results, dataset, feature_selection, classifier, iterations, labels, result_path):
    # previous very simple code that would save a csv file with the samples used in testing, their prediction probability and assigned label, for a single model iteration
    #outcome = pd.DataFrame({"sample_id": dataset[feature_selection][iteration]["x_test"].drop(["#CHROM", "POS"], axis = 1).columns, "true_label": dataset[feature_selection][iteration]["y_test"], "model_prediction": result[feature_selection][classifier][iteration]['pred_prob'], "assigned_class": result[feature_selection][classifier][iteration]['pred']})
    #outcome.to_csv(result_path + feature_selection + "/" + str(iteration) + "/" + classifier + "/predictions.csv", sep ='\t', index = False)

    labels = labels.set_index('SID')

    # Creates two dictionaries where each key is the name of a sample.
    prob = {el:[] for el in labels.index}
    lab = {el:[] for el in labels.index}

    # fills in the dictionaries with arrays that contains a prediction if the sample was used in testing
    # or "N/A" if used in training, for each iteration created by the split_dataset function
    ##### !!! this code chunk needs to be revised because I don't think its efficient !!! #####
    for i in range(0, iterations):
        for sample in dataset[feature_selection][i]["x_train"].T.index:
            prob[sample].append("N/A")
            lab[sample].append("N/A")
    for j, sample in enumerate(dataset[feature_selection][i]["x_test"].T.index):
            prob[sample].append(results[feature_selection][classifier][i]['pred_prob'][j])
            lab[sample].append(results[feature_selection][classifier][i]['pred'][j])

    # Builds dataframes from the dictionaries so that it's easier to manipulate
    prob_df = pd.DataFrame.from_dict(prob, orient='index')
    lab_df = pd.DataFrame.from_dict(lab, orient='index')

    # makes sure that each prediction is a numeric value, and not a string
    prob_df = prob_df.apply(lambda x: pd.to_numeric(x, errors='coerce'))
    lab_df = lab_df.apply(lambda x: pd.to_numeric(x, errors='coerce'))

    # 3 new columns: count of the list of predictions (each row), mean of list predictions, and the true cohort labels
    prob_df['count'] = prob_df.count(axis=1)
    prob_df['mean'] = prob_df.mean(axis=1)
    prob_df['true_label'] = pd.to_numeric(labels['GROUP'])

    # 4 new columns: count of the list of predictions (each row), mean of list predictions,
    # mode of list of predictions (assigned class from iteration voting) and the true cohort labels
    lab_df['count'] = lab_df.count(axis=1)
    lab_df['sum'] = lab_df.sum(axis=1)
    lab_df['mode'] = lab_df.mode(axis=1)
    lab_df['true_label'] = pd.to_numeric(labels['GROUP'])

    # calculate the percentage of correctly assigned classes for each test iteration
    lab_df['%Correct'] = np.nan
    lab_df.loc[(lab_df['true_label'] == 1),'%Correct'] = lab_df['sum'] / lab_df['count']
    lab_df.loc[(lab_df['true_label'] == 0),'%Correct'] = 1 - (lab_df['sum'] / lab_df['count'])

    #prob_df['true_label'] = prob_df['true_label'].apply(lambda x: bool_to_group(x))
    #lab_df['true_label'] = lab_df['true_label'].apply(lambda x: bool_to_group(x))

    # saves them as csv with tabs as the separator between items
    prob_df.to_csv(result_path + "prediction_prob.tsv", sep ='\t')
    lab_df.to_csv(result_path + "prediction_label.tsv", sep='\t')

# Function that handles the creation and saving of the ROC and CM plots
def save_results(result_path, mode, feature_selection, classifiers, iterations, results, models, datasets, labels, project_info, drug_name):
    # Making 10000 thresholds pointswfor plotting the ROC curve
    thresholds = make_thresholds(10000)
    if mode == "loo" or mode == 'cv':
        for i in feature_selection:
            for classifier in classifiers:
                result = pd.DataFrame(results[i][classifier])
                true_label = []
                pred_proba = []
                pred = []
                for j in range(0, iterations):
                    true_label.extend(result['true_label'][j])
                    pred_proba.extend(result['pred_prob'][j])
                    pred.extend(result['pred'][j])
                    if i == 'swap':
                        plot_venn_diagram(result_path + "/" + mode + "/" + i + "/" + classifier + "/", project_info, j, drug_name)
                fpr, tpr, thresholds = roc_curve(true_label, pred_proba)
                roc = roc_auc_score(true_label, pred_proba)
                result.to_csv(result_path + "/" + mode + "/" + i + "/" + classifier + "/results.tsv", sep='\t', index = False)
                mlpipeline_plot_roc(result_path + "/" + mode + "/" + i + "/" + classifier + "/", fpr, tpr, roc, classifier, i)
                save_roc(result_path + "/" + mode + "/" + i + "/" + classifier + "/", fpr, tpr, thresholds)
                mlpipeline_plot_cm(result_path + "/" + mode + "/" + i + "/" + classifier + "/", mode, models, true_label, pred, 0)

    elif mode == "cv_and_test":

        for i in feature_selection:
            for classifier in classifiers:
                # CV
                result = pd.DataFrame(results["cv"][i][classifier])
                true_label = []
                pred_proba = []
                pred = []
                for j in range(0, iterations):
                    true_label.extend(result['true_label'][j])
                    pred_proba.extend(result['pred_prob'][j])
                    pred.extend(result['pred'][j])
                    if i == 'swap':
                        plot_venn_diagram(result_path + "/" + mode + "/" + i + "/" + classifier + "/", project_info, j, drug_name)
                fpr, tpr, thresholds = roc_curve(true_label, pred_proba)
                roc = roc_auc_score(true_label, pred_proba)
                result.to_csv(result_path + "/" + mode + "/" + i + "/" + classifier + "/results.tsv", sep='\t', index = False)
                mlpipeline_plot_roc(result_path + "/" + mode + "/" + i + "/" + classifier + "/", fpr, tpr, roc, classifier, i)
                save_roc(result_path + "/" + mode + "/" + i + "/" + classifier + "/", fpr, tpr, thresholds)
                mlpipeline_plot_cm(result_path + "/" + mode + "/" + i + "/" + classifier + "/", "cv", models, true_label, pred, 0)

                #hold_out
                thresholds = make_thresholds(10000)
                result = pd.DataFrame(results["holdout"][i][classifier])
                roc = roc_auc_score(results["holdout"][i][classifier]['true_label'], results["holdout"][i][classifier]['pred_prob'])
                fpr, tpr, thresholds = roc_curve(results["holdout"][i][classifier]["true_label"], results["holdout"][i][classifier]["pred_prob"])#result = result.T
                result.to_csv(result_path + "/" + mode + "/" + i + "/" + classifier + "/hold_out/results.tsv", sep='\t', index = False)
                mlpipeline_plot_roc(result_path + "/" + mode + "/" + i + "/" + classifier + "/hold_out/", fpr, tpr, roc, classifier, i)
                save_roc(result_path + "/" + mode + "/" + i + "/" + classifier + "/hold_out/", fpr, tpr, thresholds)
                mlpipeline_plot_cm(result_path + "/" + mode + "/" + i + "/" + classifier + "/hold_out/", mode, models[i][classifier][0], results["holdout"][i][classifier]['true_label'], results["holdout"][i][classifier]['pred'], "hold_out")
                #individual_sample_report(results, datasets, i, classifier, iterations, labels, result_path + "/" + mode + "/" + i + "/" + classifier + "/")

    else:
        for i in feature_selection:
            for classifier in classifiers:
                for j in range(0, iterations):

                    mlpipeline_plot_roc(result_path + "/" + mode + "/" + i + "/" + classifier + "/" + str(j) + "/", results[i][classifier][j]['fpr'], results[i][classifier][j]['tpr'],results[i][classifier][0]['roc'], classifier, i)
                    save_roc(result_path + "/" + mode + "/" + i + "/" + classifier + "/" + str(j) + "/", results[i][classifier][j]['fpr'], results[i][classifier][j]['tpr'], thresholds)
                    mlpipeline_plot_cm(result_path + "/" + mode + "/" + i + "/" + classifier + "/" + str(j) + "/", mode, models[i][classifier][j][0], datasets[i][j]["x_test"], datasets[i][j]["y_test"], j)
                individual_sample_report(results, datasets, i, classifier, iterations, labels, result_path + "/" + mode + "/" + i + "/" + classifier + "/")
    return
