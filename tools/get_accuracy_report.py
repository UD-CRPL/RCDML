###########################################################################################################
# Tool: get_accuracy_report
# Purpose: Generates a report that provides ROC score and family information for different model results
###########################################################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from get_drug_names import get_drug_names
from pathlib import Path
import itertools
import json

# Get lists of drugs that are members of "drug_family"
def generate_drug_list(path, drug_family):
    drug_list = pd.read_excel(path + "variants_BeatAML.xlsx", sheet_name="Table S11-Drug Families", engine = "openpyxl")
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
def get_data(data_path, drug, combinations, run_info, iter):
    if run_info['hold_out']: 
        data = {fs: {classifier: pd.read_csv(data_path + "/" + str(iter) +  "/"  + drug  + "/" + run_info["date"] + run_info["validation"] + fs + "/" + classifier + "/hold_out/results.tsv" , sep="\t") for classifier in run_info["classifiers"]} for fs in run_info["fs"]}
    else:
        #print(pd.read_csv(data_path + "/" + str(iter) +  "/"  + drug  + "/" + run_info["date"] + run_info["validation"] + fs + "/" + classifier + "/results.tsv" , sep="\t"))
        data = {fs: {classifier: pd.read_csv(data_path + "/" + str(iter) +  "/"  + drug  + "/" + run_info["date"] + run_info["validation"] + fs + "/" + classifier + "/results.tsv" , sep="\t") for classifier in run_info["classifiers"]} for fs in run_info["fs"]}
    return data

def get_results(data, fs, classifier, hold_out):
    if not (hold_out):
       # print(data[fs][classifier]['true_label'])
        true = np.concatenate(data[fs][classifier]['true_label'].apply(lambda x: transform(x)).values, axis=None)
        pred = np.concatenate(data[fs][classifier]['pred'].apply(lambda x: transform(x)).values, axis=None)
        prob = np.concatenate(data[fs][classifier]['pred_prob'].apply(lambda x: transform(x)).values, axis=None)
    else:
        true = data[fs][classifier]['true_label']
        pred = data[fs][classifier]['pred']
        prob = data[fs][classifier]['pred_prob']
    tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
    sensitivity = (tp) / (tp + fn)
    specificity = (tn) / (tn + fp)
    roc = roc_auc_score(true, prob)
    return sensitivity, specificity, roc

# Using the predictions and true labels create confusion matrix, get roc score
def confusion_matrix_scores(result_path, data, drug, combinations, run_info, iter):
    all_results = pd.DataFrame()
   # for i, combo in enumerate(combinations):
    for fs in run_info['fs']:
        # gets predictions from the data
        RFsensitivity, RFspecificity, RFroc = get_results(data, fs, "rf", run_info["hold_out"])
        GBsensitivity, GBspecificity, GBroc = get_results(data,fs, 'gdb', run_info["hold_out"])
        LGBMsensitivity, LGBMspecificity, LGBMroc = get_results(data,fs, 'lgbm', run_info["hold_out"])

        # saving format
        #dict = {'Drug':drug, 'FRT':combo[0], 'Classifier':combo[1], 'sens': sensitivity, 'spec': specificity, 'auc':roc}
        dict = {'drug':drug, 'FRT':fs, 'RFsenz': RFsensitivity, 'RFspec': RFspecificity, 'RFauc':RFroc, 'GBsenz': GBsensitivity, 'GBspec': GBspecificity, 'GBauc':GBroc, 'LGBMsez':LGBMsensitivity, 'LGBMspec': LGBMspecificity, 'LGBMauc':LGBMroc}
        df = pd.DataFrame(dict, index=[0])
       # all_results = all_results.append(df)
        all_results = pd.concat([all_results, df])
        
  #      if run_info["hold_out"]:
  #          with open(result_path + "/"  + str(iter) +  "/" + drug + "/" + run_info["date"] + run_info["validation"] + combo[0] + "/" + combo[1] + '/hold_out/accuracy_report.txt', 'w') as file:
  #              file.write(json.dumps(dict))
  #      else:
  #          with open(result_path + "/"  + str(iter) +  "/" + drug + "/" + run_info["date"] + run_info["validation"] + combo[0] + "/" + combo[1] + '/accuracy_report.txt', 'w') as file:
  #              file.write(json.dumps(dict))
    return all_results

# Gets list of damilies
def get_drug_families(data_path):
    drug_list = pd.read_excel(data_path + "variants_BeatAML.xlsx", sheet_name="Table S11-Drug Families", engine = "openpyxl")
    drug_list["inhibitor"] = drug_list["inhibitor"].apply(lambda x: x.split(" ")[0])
    drug_list = drug_list.groupby("inhibitor")["family"].apply(', '.join).reset_index()
    return drug_list

def get_missing(path, drug_list, combination, iterations, run_info):
    ismissing = False
    for combo in combination:
        print("Combo: " , combo[0] + " + " + combo[1])
        for i in range(1, iterations + 1):
            print("Iteration: ", i)
            missing = []
            for drug in drug_list:
                if run_info["hold_out"]:
                    if not Path(path  + "/" + str(i) +  "/" + drug + "/" + run_info["date"] + run_info["validation"] + combo[0] + "/" + combo[1] + "/hold_out/results.tsv").exists():
                        missing.append(drug)
                        ismissing = True
                else:
                   # print(path  + "/" + str(i) +  "/" + drug + "/" + run_info["date"] + run_info["validation"] + combo[0] + "/" + combo[1] + "/hold_out/")
                    if not Path(path  + "/" + str(i) +  "/" + drug + "/" + run_info["date"] + run_info["validation"] + combo[0] + "/" + combo[1] + "/results.tsv").exists():
                        missing.append(drug)
                        ismissing = True
            print("Missing Drugs: ", missing)
    return ismissing

def make_result_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

iterations = 10
run_name = "/pr_results/unbalanced/"
data_path = "/mnt/d/school_and_work/target/results/target/" + run_name
#drug_list = generate_drug_list('/Users/mf0082/Documents/Nemours/AML/beatAML/dataset/', 'RTK_TYPE_III')
#drug_list = ["Crenolanib","Dasatinib", "Foretinib", "Regorafenib"]
#drug_list = list(get_drug_names("/Users/mf0082/Documents/Nemours/AML/beatAML/dataset/", "RTK_TYPE_III")[0])
drug_list = ["Sorafenib"]
feature_selection = ['pca', 'shap', 'dge']
classifiers = ["rf", 'gdb', 'lgbm']
validation = "/cv_and_test/"
date = "/06-08-2023/"
hold_out = False

run_info = {"fs":feature_selection, "classifiers":classifiers, "validation":validation, "date":date, "hold_out": hold_out}
combinations = generate_combinations(run_info)
if not get_missing(data_path, drug_list, combinations, iterations, run_info):
    result_dir = data_path + "/all_results/"
    if hold_out:
        result_dir = result_dir + "/hold_out/"      
    else:
        result_dir = result_dir + "/cv/"
    make_result_dir(result_dir)
    writer = pd.ExcelWriter(result_dir + '/all_results_random.xlsx', engine='openpyxl')
    final_results = []
    for iteration in range(1, iterations + 1):
        all_results = pd.DataFrame()
        for drug in drug_list:
            if not hold_out:
                data = get_data(data_path, drug, combinations, run_info, iteration)
                scores = confusion_matrix_scores(data_path, data, drug, combinations, run_info, iteration)
                all_results = pd.concat([all_results, scores])
            else: 
                data = get_data(data_path, drug, combinations, run_info, iteration)
                scores = confusion_matrix_scores(data_path, data, drug, combinations, run_info, iteration)
                all_results = pd.concat([all_results, scores])
            all_results.to_excel(writer, sheet_name='Iteration ' + str(iteration), index = False)
            final_results.append(all_results)
   # print(final_results)
    final_results = pd.concat(final_results).groupby(['drug', 'FRT']).mean()
    print(final_results)
    final_results.to_csv(result_dir + '/avg_results_random.txt', sep="\t")
    final_results.to_excel(writer, sheet_name='Average', index = True)  
    writer.save()
else:
    print("Missing results, no report generated")
