###########################################################################################################
# Tool: get_combined_roc
# Purpose: Used to create ROC Curve Plot for model comparisons
###########################################################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

def factorize(num):
    return [n for n in range(1, num + 1) if num % n == 0]


def find_middle(input_list):
    middle = float(len(input_list)) / 2
    if len(input_list) % 2 != 0:
        return (input_list[int(middle - .5)], input_list[int(middle - .5)])
    else:
        return (input_list[int(middle)], input_list[int(middle - 1)])

def get_subplots_gridsize(num):
    factors = factorize(num)
    middle = find_middle(factors)
    return middle

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

def roc_all_drugs(data_path, drug_list, run_info):
    class_names = ['Low Resp','High Resp']
    np.random.seed(667)
    number_of_plots = len(drug_list)
    #print(number_of_plots)
    gridsize = get_subplots_gridsize(number_of_plots)
    #print(gridsize)
#    colors = [np.random.rand(3,) for i in range(0, len(run_info["classifiers"]) + len(run_info["fs"]))]

    data = {drug: get_data(data_path, drug, run_info) for drug in drug_list}
    print(data)

    for i_fs, fs in enumerate(run_info["fs"]):
        for i_class, classifier in enumerate(run_info["classifiers"]):
            fig, axis = plt.subplots(gridsize[0], gridsize[1], sharex=True, sharey=True)
            k = 0
            for i in range(0, gridsize[0]):
                for j in range(0, gridsize[1]):
                    if not run_info["hold_out"]:          # print(data[fs][classifier]['true_label'])
                        true = np.concatenate(data[drug_list[k]][fs][classifier]['true_label'].apply(lambda x: transform(x)).values, axis=None)
                        pred = np.concatenate(data[drug_list[k]][fs][classifier]['pred'].apply(lambda x: transform(x)).values, axis=None)
                        prob = np.concatenate(data[drug_list[k]][fs][classifier]['pred_prob'].apply(lambda x: transform(x)).values, axis=None)
                    else:
                        true = data[drug_list[k]][fs][classifier]['true_label']
                        pred = data[drug_list[k]][fs][classifier]['pred']
                        prob = data[drug_list[k]][fs][classifier]['pred_prob']
                    cm = confusion_matrix(pred, true, labels=[0, 1])
                    disp = ConfusionMatrixDisplay.from_predictions(pred, true, display_labels=class_names, ax = axis[i, j], cmap=plt.get_cmap("YlGnBu"), colorbar = False)
                    #axis[i, j] = disp.set(xlabel='', ylabel='', ax = axis[i, j])
                    title = drug_list[k]
                    axis[i, j].set_title(title)
                    axis[i, j].set_xlabel('')
                    axis[i, j].set_ylabel('')
                    axis[i, j].set_yticklabels(class_names, va = 'center', rotation = 'vertical')
                    k = k + 1
                    
            for label, ax in enumerate(axis.flat):
              #ax.set_title('Normal Title', fontstyle='italic')
                #ax.set_title(abc[label] + ")", fontfamily='serif', loc='left', fontsize='medium')
                 ax.set_title(str(label + 1) + ")", fontfamily='serif', loc='left', fontsize='medium')

            fig.text(0.5, 0.025, 'Predicted Label', ha='center', fontsize = 12)
            fig.text(0.065, 0.5, 'True Label', va='center', rotation='vertical', fontsize = 12)

            #leg = plt.legend(bbox_to_anchor=(2.05, 0.025), loc='lower right')
            #for lh in leg.legendHandles: 
            #    lh.set_alpha(1)
            #plt.suptitle("CM for " + str(number_of_plots) +  " RTK_TYPE_III Inhibitors: " + fs.upper() + " + " + classifier.upper())
            #plt.tight_layout()
           # plt.savefig(data_path + "/" + fs + "_" + classifier + "_cm.png")
            fig.set_size_inches(12, 15)
            fig.set_dpi(1200)
            plt.savefig(data_path + "/" + fs + "_" + classifier + "_cm.tif")
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
    if run_info['hold_out']: 
        data = {fs: {classifier: pd.read_csv(data_path +  "/"  + drug  + "/" + run_info["date"] + run_info["mode"] + fs + "/" + classifier + "/hold_out/results.tsv" , sep="\t") for classifier in run_info["classifiers"]} for fs in run_info["fs"]}
    else:
        #print(pd.read_csv(data_path + "/" + str(iter) +  "/"  + drug  + "/" + run_info["date"] + run_info["validation"] + fs + "/" + classifier + "/results.tsv" , sep="\t"))
        data = {fs: {classifier: pd.read_csv(data_path +  "/"  + drug  + "/" + run_info["date"] + run_info["mode"] + fs + "/" + classifier + "/results.tsv" , sep="\t") for classifier in run_info["classifiers"]} for fs in run_info["fs"]}
    return data

def get_roc_xy(path):
    df = pd.read_csv(path + "/roc_data.tsv" , sep="\t")
    return {"x": df["sensitivity"], "y": df["1 - specificity"]}

def make_result_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

iterations = 1
run_name = "30features_25split"
data_path = "/Users/mf0082/Documents/Bioinformatics_paper/results/beatAML/new_results/" + run_name
result_path = "/Users/mf0082/Documents/Bioinformatics_paper/results/beatAML/new_results/" + run_name 
feature_selection = ["shap","pca","dge","random"]
classifiers = ["rf", "gdb"]
mode = "/cv_and_test/"
date = "/07-29-2022/"
hold_out = True
#drug_list = ["Crenolanib", "Foretinib", "Regorafenib", "Dasatinib"]
drug_list = generate_drug_list("/Users/mf0082/Documents/Nemours/AML/beatAML/dataset/", "RTK_TYPE_III")
run_info = {"fs":feature_selection, "classifiers":classifiers, "mode":mode, "date":date, "hold_out": hold_out}
make_result_dir(result_path)

for iteration in range(1, iterations + 1):
  #  print("ROC Plot for Iteration: ", iteration)
    #for drug in drug_list:
      #  data = get_data(data_path + "/" + str(iteration) + "/", drug, run_info)
      #  mlpipeline_plot_roc(result_path +  "/" + str(iteration) + "/", data, drug, run_info)
    roc_all_drugs(data_path +  "/" + str(iteration) + "/", drug_list, run_info)
