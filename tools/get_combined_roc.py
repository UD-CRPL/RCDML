###########################################################################################################
# Tool: get_combined_roc
# Purpose: Used to create ROC Curve Plot for model comparisons
###########################################################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import auc
from pathlib import Path
from get_drug_names import get_drug_names

def factorize(num):
    return [n for n in range(1, num + 1) if num % n == 0]


def find_middle(input_list):
    middle = float(len(input_list)) / 2
    print(middle)
    if len(input_list) % 2 != 0:
        return int(middle - .5), int(middle - .5)
    else:
        return int(middle), int(middle)

def get_subplots_gridsize(num):
    factors = factorize(num)
    middle = find_middle(factors)
    return middle

def roc_all_drugs(data_path, drug_list, run_info):
    np.random.seed(667)
    number_of_plots = len(drug_list)
    print(number_of_plots)
    gridsize = find_middle(drug_list)
    print(gridsize)
    fig, axis = plt.subplots(gridsize[0], gridsize[1], sharex=True, sharey=True)
    colors = mcolors.TABLEAU_COLORS
    colors = list(colors)
    k = 0
    for i in range(0, gridsize[0]):
        for j in range(0, gridsize[1]):
            axis[i, j].plot([0, 1], [0, 1], color='darkblue', linestyle='--')
            data = get_data(data_path, drug_list[k], run_info)
            for i_fs, fs in enumerate(run_info["fs"]):
                for i_class, classifier in enumerate(run_info["classifiers"]):
                    auc_score = auc(data[fs][classifier]["x"], data[fs][classifier]["y"])
                    axis[i, j].plot(data[fs][classifier]["x"], data[fs][classifier]["y"], color=colors[i_fs + i_class], label= (fs.upper() + " + " + classifier.upper()), alpha = .6)
            axis[i, j].set(xlabel='', ylabel='')
            title = drug_list[k]
            axis[i, j].set_title(title)
            k = i + j

    for label, ax in enumerate(axis.flat):
        #ax.set_title('Normal Title', fontstyle='italic')
        #ax.set_title(abc[label] + ")", fontfamily='serif', loc='left', fontsize='medium')
        ax.set_title(str(label + 1) + ")", fontfamily='serif', loc='left', fontsize='medium')
    
    for ax in axis.flat:
    #    ax.set(xlabel='1 - Specificity (False Positive Rate)', ylabel='Sensitivity (True Positive Rate)')
        ax.label_outer()

    fig.text(0.5, 0.025, '1 - Specificity (False Positive Rate)', ha='center', fontsize = 12)
    fig.text(0.04, 0.5, 'Sensitivity (True Positive Rate)', va='center', rotation='vertical', fontsize = 12)

    leg = plt.legend(bbox_to_anchor=(1.55, 0.025), loc='lower right')
    for lh in leg.legendHandles: 
        lh.set_alpha(1)
    #plt.tight_layout()
    fig.set_size_inches(12.5, 12.5)
    fig.set_dpi(1200)
    plt.savefig(data_path + "_roc.tif")
    #plt.show()
    plt.clf()

# Get list of drugs that are part of the specified "drug_family"
def generate_drug_list(path, drug_family):
    drug_list = pd.read_excel(path + "variants_BeatAML.xlsx", sheet_name="Table S11-Drug Families", engine ="openpyxl")
    drug_list = drug_list[drug_list["family"] == drug_family]
    drug_list = drug_list['inhibitor']
    if drug_family == 'RTK_TYPE_III':
        invalid_drugs = [drug_list[drug_list == 'Tandutinib (MLN518)'].index[0]]
        drug_list = drug_list.drop(invalid_drugs)
    drug_list = drug_list.apply(lambda x: x.split(' ')[0])
    drug_list = drug_list.values
    return drug_list

# Plots the ROC curves based on the data points gathered from get_data
def mlpipeline_plot_roc(result_path, data, drug, run_info):
    np.random.seed(667)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    for fs in run_info["fs"]:
        for classifier in run_info["classifiers"]:
            auc_score = auc(data[fs][classifier]["x"], data[fs][classifier]["y"])
            plt.plot(data[fs][classifier]["x"], data[fs][classifier]["y"], color=np.random.rand(3,), label= fs.upper() + " + " + classifier.upper() + ' auc=' + str(round(auc_score, 3)), alpha = .6)
    plt.xlabel('1 - Specificity (False Positive Rate)')
    plt.ylabel('Sensitivity (True Positive Rate)')
    plt.legend()
    plt.title('ROC Model: ' + drug.upper())
    plt.savefig(result_path + "/" + drug + "/combined_roc.png")
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
def get_data(data_path, drug, run_info):
    data = {fs: {classifier: get_roc_xy(data_path + drug + run_info["date"] + run_info["mode"] + fs + "/" + classifier) for classifier in run_info["classifiers"]} for fs in run_info["fs"]}       
    return data

def get_roc_xy(path):
    df = pd.read_csv(path + "/roc_data.tsv" , sep="\t")
    return {"x": df["sensitivity"], "y": df["1 - specificity"]}

def make_result_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

iterations = 10
run_name = "pi3k-akt-mtor"
data_path = "/Users/mf0082/Documents/Bioinformatics_paper/results/beatAML/results_1014/" + run_name
result_path = "/Users/mf0082/Documents/Bioinformatics_paper/results/beatAML/results_1014/" + run_name 
feature_selection = ["shap","pca","dge","random"]
classifiers = ["rf", "gdb"]
mode = "/cv_and_test/"
date = "/10-11-2022/"
hold_out = True
#drug_list = ["Cediranib", "Sorafenib", "Regorafenib", "Dasatinib"]
#drug_list = generate_drug_list("/Users/mf0082/Documents/Nemours/AML/beatAML/dataset/", "BTK_TEC")
drug_list = list(get_drug_names("/Users/mf0082/Documents/Nemours/AML/beatAML/dataset/", "PI3K-AKT_MTOR")[0])
run_info = {"fs":feature_selection, "classifiers":classifiers, "mode":mode, "date":date, "hold_out": hold_out}
make_result_dir(result_path)

for iteration in range(1, iterations + 1):
 #   print("ROC Plot for Iteration: ", iteration)
 #   for drug in drug_list:
 #       data = get_data(data_path + "/" + str(iteration) + "/", drug, run_info)
 #       mlpipeline_plot_roc(result_path +  "/" + str(iteration) + "/", data, drug, run_info)
    roc_all_drugs(data_path +  "/" + str(iteration) + "/", drug_list, run_info)
