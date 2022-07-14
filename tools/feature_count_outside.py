###########################################################################################################
# Tool: feature_count
# Purpose: Generates a feature counter from given result path. Counts how many times the feature was picked
# in the feature selection setup.
###########################################################################################################
import pandas as pd
import numpy as np
from pathlib import Path
import os

def make_result_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

# Given a drug family, generate a list of all the drugs that are part of the family
def generate_drug_list(path, drug_family):
    drug_list = pd.read_excel(path + "aml/beatAML/variants_BeatAML.xlsx", sheet_name="Table S11-Drug Families")
    drug_list = drug_list[drug_list["family"] == drug_family]
    drug_list = drug_list['inhibitor']
    if drug_family == 'RTK_TYPE_III':
        invalid_drugs = [drug_list[drug_list == 'Tandutinib (MLN518)'].index[0], drug_list[drug_list == "PD173955"].index[0]]
        drug_list = drug_list.drop(invalid_drugs)
    drug_list = drug_list.apply(lambda x: x.split(' ')[0])
    return drug_list

def build_feature_counter(features):
    dict = {feature:0 for feature in features}
    return dict

def add_to_feature_counter(features, counter):
    for feature in features:
        print(features)
        print(feature)
        counter[feature] = counter[feature] + 1
    return

def get_features(path, drug, counter, dict):
    for i in range(1, dict["iterations"] + 1):
        print("GETTING FEATURES FOR DRUG: "  + drug + " FROM ITERATION: " + str(i))
        data = pd.read_csv(path + str(i) + "/feature_counters/" + "/" + drug + "/" + drug + "_shap_feature_counter.tsv", sep='\t')
        data = data['GENE ID'].values
        print(data)
        add_to_feature_counter(data, counter)
    return

features_path = "/Users/mf0082/Documents/Nature_Comm_paper/Code/results/beatAML/bioinformatics_new/"
dataset_path = "/Users/mf0082/Documents/MODEL/dataset/"
drug_list = generate_drug_list(dataset_path, "RTK_TYPE_III")
#drug_list = ["Regorafenib", "Crenolanib", "Dasatinib", "KW-2449", "Foretinib"]
mode = "/cv_and_test/"
date = "/05-31-2022/"
feature_selection = "/shap/"
classifier = "/rf/"
iterations = 10
dict = {"fs":feature_selection, "classifier":classifier, "mode":mode, "date":date, "iterations": iterations}
result = None
result_path = features_path
# Loads the RNA Sequence BeatAML dataset and gets the list of genes
gene_list = pd.read_excel(dataset_path + "aml/beatAML/variants_BeatAML.xlsx", sheet_name="Table S9-Gene Counts CPM")
gene_list = gene_list['Gene']
print("finished loading the gene list")

for i, drug in enumerate(drug_list):
    print(drug + ":")
    counter = build_feature_counter(gene_list)
    get_features(features_path, drug, counter, dict)
    counter = {k: v for k, v in sorted(counter.items(), key=lambda item: item[1])}
    counter_df = pd.DataFrame(counter.items(), columns=["GENE ID", "FREQUENCY"])
    counter_df = counter_df[counter_df["FREQUENCY"] != 0]
    counter_df = counter_df.sort_values(by=['FREQUENCY'], ascending=False)
    #counter_df["DRUG ID"] = drug
    make_result_dir(result_path + "/genes_selected/cv/" + drug)
    counter_df.to_csv(result_path + "/genes_selected/cv/" + drug + "/" + drug + "_shap_feature_counter.tsv", index = False, sep="\t")

    #if i == 0:
    #    result = counter_df
    #else:
    #    result = pd.concat([counter_df, result])

#result.to_csv(features_path + "/Regorafenib_random_feature_counter.tsv", index = False, sep="\t")
