import get_combined_cm as pcm 
import get_combined_roc as proc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import auc
from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

def make_roc_cm_sc22_plot(data_path, drug_list, run_info):
    class_names = ['Negative Cohort','Positive Cohort']
    np.random.seed(3000)
    colors = [np.random.rand(3,) for i in range(0, len(run_info["classifiers"]) + len(run_info["fs"]))]
    colors = mcolors.TABLEAU_COLORS
    colors = list(colors)
    cm_data = {drug: pcm.get_data(data_path, drug, run_info) for drug in drug_list}
    roc_data = {drug: proc.get_data(data_path, drug, run_info) for drug in drug_list}

    fig, axis = plt.subplots(2, 4)
    for i in range(0, 2):
        k = 0
        for j in range(0, 4):
            if i == 0:
                # ROC
                axis[i, j].plot([0, 1], [0, 1], color='darkblue', linestyle='--')
                for i_fs, fs in enumerate(run_info["fs"]):
                    for i_class, classifier in enumerate(run_info["classifiers"]):
                        auc_score = auc(roc_data[drug_list[k]][fs][classifier]["x"], roc_data[drug_list[k]][fs][classifier]["y"])
                        axis[i, j].plot(roc_data[drug_list[k]][fs][classifier]["x"], roc_data[drug_list[k]][fs][classifier]["y"], color=colors[i_fs + i_class], label= fs.upper() + " + " + classifier.upper(), alpha = .6)
                axis[i, j].set_title(drug_list[k].upper())
                axis[i, j].set_xlabel("FPR")
                axis[i, j].set_ylabel("TPR")
            else:
                # CM
                true = np.concatenate(cm_data[drug_list[k]]['shap']['rf']['true_label'].apply(lambda x: pcm.transform(x)).values, axis=None)
                pred = np.concatenate(cm_data[drug_list[k]]['shap']['rf']['pred'].apply(lambda x: pcm.transform(x)).values, axis=None)
                prob = np.concatenate(cm_data[drug_list[k]]['shap']['rf']['pred_prob'].apply(lambda x: pcm.transform(x)).values, axis=None)
                cm = confusion_matrix(pred, true, labels=[0, 1])
                disp = ConfusionMatrixDisplay.from_predictions(pred, true, display_labels=class_names, ax = axis[i, j], cmap=plt.get_cmap("YlGnBu"), colorbar = False)
                axis[i, j].set_yticklabels(class_names, va = 'center', rotation = 'vertical')
            k = k + 1

    leg = axis[0, 3].legend(bbox_to_anchor=(1.55, 0.025), loc='lower right')
    plt.show()

iterations = 1
run_name = "sc22_results"
data_path = "/Users/mf0082/Documents/Bioinformatics_paper/results/beatAML/" + run_name
result_path = "/Users/mf0082/Documents/Bioinformatics_paper/results/beatAML/" + run_name 
feature_selection = ["shap","pca","dge","random"]
classifiers = ["rf", "gdb"]
mode = "/cv/"
date = "/08-05-2022/"
hold_out = False
drug_list = ["Crenolanib", "Foretinib", "Regorafenib", "Dasatinib"]
run_info = {"fs":feature_selection, "classifiers":classifiers, "mode":mode, "date":date, "hold_out": hold_out}
pcm.make_result_dir(result_path)

for iteration in range(1, iterations + 1):
    make_roc_cm_sc22_plot(data_path +  "/" + str(iteration) + "/", drug_list, run_info)
    