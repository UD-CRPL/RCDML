import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('/home/mferrato/RCDML/')
import data_preprocess as dp
from pathlib import Path

result_path = "/mnt/d/school_and_work/BeatAML/results/"
dataset_path = "/mnt/d/school_and_work/BeatAML/dataset/"
result_name = "/cluster_heatmaps/"
drug_list = ["Dasatinib", "Dovitinib","Foretinib", "KW-2449","Sorafenib"]
#make_result_dir(result_path)


def make_result_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

    
def hierarchical_clustering_heatmap(path, drug, data, labels):   
    sys.setrecursionlimit(100000)
    data = data.T
    lut = {0: 'b', 1: 'g'}
    rows = labels[labels['SID'].isin(data.index.values)].set_index('SID')
    row_colors = rows["GROUP"].map(lut)
#    col_colors = species.map(lut2)
    heatmap = sns.clustermap(data, standard_scale=1, cmap='Blues', row_colors=row_colors, figsize=(100, 100))
    heatmap.savefig(path + drug + "_heatmap_clustering.png")
    plt.clf()
    return

def load_features(path, drug):
    features = pd.read_csv(path + "SHAP_genes_selected.csv", usecols=["GENE ID", "DRUG ID"], sep=",")
    features = features[features["DRUG ID"] == drug]
    return features

data, samples = dp.load_dataset_beatAML(dataset_path, "cpm")

make_result_dir(result_path + result_name)

for drug in drug_list:
    labels = dp.load_labels_beatAML(dataset_path, drug)
    matched_data, matched_labels, matched_samples = dp.sample_match(data, labels, samples)
    features = load_features(dataset_path, drug)
    filtered_data = matched_data.loc[features["GENE ID"]]
    hierarchical_clustering_heatmap(result_path + result_name, drug, filtered_data, matched_labels)
    #print(features)
    #print(matched_labels)