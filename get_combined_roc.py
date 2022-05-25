###########################################################################################################
# Tool: get_combined_roc
# Purpose: Used to create ROC Curve Plot for model comparisons
###########################################################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from pathlib import Path

# Get list of drugs that are part of the specified "drug_family"
def generate_drug_list(path, drug_family):
    drug_list = pd.read_excel(path + "aml/beatAML/variants_BeatAML.xlsx", sheet_name="Table S11-Drug Families")
    drug_list = drug_list[drug_list["family"] == drug_family]
    drug_list = drug_list['inhibitor']
    if drug_family == 'RTK_TYPE_III':
        invalid_drugs = [drug_list[drug_list == 'Tandutinib (MLN518)'].index[0], drug_list[drug_list == "PD173955"].index[0]]
        drug_list = drug_list.drop(invalid_drugs)
    drug_list = drug_list.apply(lambda x: x.split(' ')[0])
    return drug_list

# Plots the ROC curves based on the data points gathered from get_data
def mlpipeline_plot_roc(result_path, data, drug, combinations):
    np.random.seed(667)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    for i, combo in enumerate(combinations):
        auc_score = auc(data[i][0], data[i][1])
        plt.plot(data[i][0], data[i][1], color=np.random.rand(3,), label= combo[0].upper() + " + " + combo[1].upper() + ' auc=' + str(round(auc_score, 3)), alpha = .6)
    plt.xlabel('1 - Specificity (False Positive Rate)')
    plt.ylabel('Sensitivity (True Positive Rate)')
    plt.legend()
    plt.title('ROC Model: ' + drug.upper())
    plt.savefig(result_path + drug + ".png")
    plt.clf()
    return

# Creates all feature selection + classifier combos for iteration purposes
def generate_combinations(dict):
    combinations = []
    for fs in dict["fs"]:
        for classifier in dict["classifiers"]:
            combinations.append((fs, classifier))
    return combinations

# Gets the ROC data points
def get_data(data_path, drug, combinations, dict):
    data = []
    for combo in combinations:
        df = pd.read_csv(data_path + drug + dict["date"] + dict["mode"] + combo[0] + "/" + combo[1] + "/roc_data.tsv" , sep="\t")
        data.append((df["sensitivity"], df["1 - specificity"]))
    return data

def make_result_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

data_path = "/Users/mf0082/Documents/Nature_Comm_paper/Code/beatAML/test/"
result_path = "/Users/mf0082/Documents/Nature_Comm_paper/Code/beatAML/test/roc/"
feature_selection = ["shap","pca","dge","none","random"]
classifiers = ["rf"]
mode = "/cv/"
date = "/02-17-2022/"
dict = {"fs":feature_selection, "classifiers":classifiers, "mode":mode, "date":date}
drug_list = ["Cediranib"]
make_result_dir(result_path)

combinations = generate_combinations(dict)

for drug in drug_list:
    data = get_data(data_path, drug, combinations, dict)
    mlpipeline_plot_roc(result_path, data, drug, combinations)
