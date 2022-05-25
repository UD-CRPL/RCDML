###########################################################################################################
# Tool: get_accuracy_report
# Purpose: Generates a report that provides ROC score and family information for different model results
###########################################################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from pathlib import Path
import itertools
import json

# Get lists of drugs that are members of "drug_family"
def generate_drug_list(path, drug_family):
    drug_list = pd.read_excel(path + "aml/beatAML/variants_BeatAML.xlsx", sheet_name="Table S11-Drug Families")
    drug_list = drug_list[drug_list["family"] == drug_family]
    drug_list = drug_list['inhibitor']
    if drug_family == 'RTK_TYPE_III':
        invalid_drugs = [drug_list[drug_list == 'Tandutinib (MLN518)'].index[0], drug_list[drug_list == "PD173955"].index[0]]
        drug_list = drug_list.drop(invalid_drugs)
    drug_list = drug_list.apply(lambda x: x.split(' ')[0])
    return drug_list

#  Creates list of all feature selection + classifier combinations
def generate_combinations(dict):
    combinations = []
    for fs in dict["fs"]:
        for classifier in dict["classifiers"]:
            combinations.append((fs, classifier))
    return combinations

# Used to parse through the results file and recover the predictions
def transform(x):
    result = x.replace("]","")
    result = result.replace("[","")
    result = result.replace(",","")
    result = result.replace("\n", "")
    result = result.split(" ")
    result = list(filter(None, result))
    for i in range(0, len(result)):
        result[i] = float(result[i])
    result = np.concatenate(result, axis=None)
    return result

# Gets all the result data and puts it in dataframes
def get_data(data_path, drug, combinations, dict, swap = None):
    data = []
    for combo in combinations:
        if not swap:
            df = pd.read_csv(data_path + "/" + drug + dict["date"] + dict["validation"] + combo[0] + "/" + combo[1] + "/results.tsv" , sep="\t")
            data.append(df)
        else:
            df = pd.read_csv(data_path + "/" + swap + "/" + drug + dict["date"] + dict["validation"] + combo[0] + "/" + combo[1] + "/results.tsv" , sep="\t")
            data.append(df)
    return data

# Using the predictions and true labels create confusion matrix, get roc score
def confusion_matrix_scores(result_path, data, drug, combinations, family, run_info, swap = None):
    for i, combo in enumerate(combinations):
        # gets predictions from the data
        true = np.concatenate(data[i]['true_label'].apply(lambda x: transform(x)).values, axis=None)
        pred = np.concatenate(data[i]['pred'].apply(lambda x: transform(x)).values, axis=None)
        prob = np.concatenate(data[i]['pred_prob'].apply(lambda x: transform(x)).values, axis=None)

        tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
        sensitivity = (tp) / (tp + fn)
        specificity = (tn) / (tn + fp)
        roc = roc_auc_score(true, prob)

        # saving format
        if not swap:
            dict = {'Drug':drug, 'Drug Family':family[family['inhibitor'] == drug]['family'].values[0], 'Feature Selection':combo[0], 'Classifier':combo[1], 'ROC-AUC':roc}
            df = pd.DataFrame(dict, index=[0])
            with open(result_path + "/" + drug + run_info["date"] + run_info["validation"] + combo[0] + "/" + combo[1] + '/accuracy_report.txt', 'w') as file:
                file.write(json.dumps(dict))
        else:
            dict = {'Original':drug, 'Original Family':family[family['inhibitor'] == drug]['family'].values[0], 'Swapped':swap, 'Swapped Family':family[family['inhibitor'] == swap]['family'].values[0], 'Sensitivity':sensitivity, 'Specificity':specificity, 'ROC-AUC':roc}
            df = pd.DataFrame(dict, index=[0])
            with open(result_path + "/" + swap + "/" + drug + run_info["date"] + run_info["validation"] + combo[0] + "/" + combo[1] + '/result_scores.txt', 'w') as file:
                file.write(json.dumps(dict))
    return df

# Gets list of damilies
def get_drug_families(data_path):
    drug_list = pd.read_excel(data_path + "aml/beatAML/variants_BeatAML.xlsx", sheet_name="Table S11-Drug Families")
    drug_list["inhibitor"] = drug_list["inhibitor"].apply(lambda x: x.split(" ")[0])
    drug_list = drug_list.groupby("inhibitor")["family"].apply(', '.join).reset_index()
    return drug_list

data_path = "/Users/mf0082/Documents/Nature_Comm_paper/Code/results/beatAML/test/"
drug_list = ['Cediranib']
swapped = ['Sorafenib']
feature_selection = ['swap','random','dge','pca']
classifiers = ["rf",'gdb']
validation = "/cv/"
date = "/02-28-2022/"
family = get_drug_families('/Users/mf0082/Documents/MODEL/dataset/')

dict = {"fs":feature_selection, "classifiers":classifiers, "validation":validation, "date":date}
combinations = generate_combinations(dict)
all_results = pd.DataFrame()
for drug in drug_list:
    if not swapped:
        data = get_data(data_path, drug, combinations, dict)
        scores = confusion_matrix_scores(data_path, data, drug, combinations, family, dict)
        all_results = all_results.append(scores)
    else:
        for swap in swapped:
            data = get_data(data_path, drug, combinations, dict, swap)
            scores = confusion_matrix_scores(data_path, data, drug, combinations, family, dict, swap)
            all_results = all_results.append(scores)

print(all_results.reset_index(drop=True))
