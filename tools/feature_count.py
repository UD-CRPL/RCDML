###########################################################################################################
# Tool: feature_count
# Purpose: Generates a feature counter from given result path. Counts how many times the feature was picked
# in the feature selection setup.
###########################################################################################################
import pandas as pd
import numpy as np
from pathlib import Path
import os

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
        counter[feature[0]] = counter[feature[0]] + 1
    return

def get_features(path, drug, counter, dict):
    for i in range(0, dict["iterations"]):
        add_to_feature_counter(pd.read_csv(path + drug + dict["date"] + dict["mode"] + dict["fs"] + dict["classifier"] + str(i) + "/genes_selected.tsv", sep='\t', header=None).values, counter)
    return

features_path = "/Users/mf0082/Documents/Nature_Comm_paper/Code/beatAML/test/"
dataset_path = "/Users/mf0082/Documents/MODEL/dataset/"
#drug_list = generate_drug_list(dataset_path, "RTK_TYPE_III")
drug_list = ["Cediranib"]
mode = "/cv/"
date = "/02-17-2022/"
feature_selection = "/shap/"
classifier = "/rf/"
iterations = 3
dict = {"fs":feature_selection, "classifier":classifier, "mode":mode, "date":date, "iterations": iterations}
result = None

# Loads the RNA Sequence BeatAML dataset and gets the list of genes
gene_list = pd.read_excel(dataset_path + "aml/beatAML/variants_BeatAML.xlsx", sheet_name="Table S9-Gene Counts CPM")
gene_list = gene_list['Gene']
print("finished loading the gene list")

for i, drug in enumerate(drug_list):
    counter = build_feature_counter(gene_list)
    get_features(features_path, drug, counter, dict)
    counter = {k: v for k, v in sorted(counter.items(), key=lambda item: item[1])}
    counter_df = pd.DataFrame(counter.items(), columns=["GENE ID", "FREQUENCY"])
    counter_df = counter_df[counter_df["FREQUENCY"] != 0]
    counter_df = counter_df.sort_values(by=['FREQUENCY'], ascending=False)
    counter_df["DRUG ID"] = drug
    if i == 0:
        result = counter_df
    else:
        result = pd.concat([counter_df, result])

result.to_csv(features_path + "/feature_counter.tsv", index = False, sep="\t")
