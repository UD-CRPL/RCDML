import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from collections import Counter
import argparse


parser = argparse.ArgumentParser(description='Parser for limma util tool')
parser.add_argument("--dataset")
parser.add_argument("--drug")
parser.add_argument("--dir")
args = parser.parse_args()

#path = "/Users/mf0082/Documents/Nemours/AML/beatAML/"
#limma_path = path + "LIMMA/"
#url = path + "dataset/"
#results_path = path + "results/new_limma/"
url = args.dataset
limma_path = args.dir
results_path = args.dir
drug_name  = args.drug
shap_path_1 = "/Users/mf0082/Documents/MODEL/results/beatAML/real_run/"
shap_path_2 = "/11-28-2021/cross_validation/shap/rf/"

k = 5

def plot_venn_diagram(path, drug, limma, shap, iteration):
    A = set(limma['DGE'])
    B = set(shap['SHAP'])
    #print(A)
    AB_overlap = A & B  #compute intersection of set A & set B
    #text = [A, B, AB_overlap]
    sets = Counter()           #set order A, B
    sets['10'] = len(A-AB_overlap) #10 denotes A on, B off
    sets['01'] = len(B-AB_overlap) #01 denotes A off, B on
    sets['11'] = len(AB_overlap)   #11 denotes A on, B on
    plt.figure(figsize=(7,7))
    ax = plt.gca()
    v = venn2(subsets=sets, set_labels=["DGE", "SHAP"], ax=ax, set_colors=('red','blue'),alpha=0.5)
    plt.title('Overlapping features for Iteration - ' + str(iteration) + " : " + drug)
    #positions = ['10', '01', '11']
    #for i in range(0, len(positions)):
    #    v.get_label_by_id(positions[i]).set_text(text[i])\
    #overlap = pd.Series([x for x in real_set if x in flipped_set], name='Overlap')
    #df = pd.concat([real_set, flipped_set, overlap], axis=1)
    #df = pd.DataFrame(data={labels[0]:feature_sets[0], labels[1]:feature_sets[1], 'Overlap':overlap})
    venn_path = path + "/venn_d/" + drug + "/" + str(iteration)
    make_result_dir(venn_path)
    #df.to_csv(venn_path + "/overlapping_genes.tsv", sep='\t', index=False)
    with open(venn_path + "/overlapping_genes.tsv", 'w') as file:
        for row in list(AB_overlap):
            s = "".join(map(str, row))
            file.write(s+'\n')
    plt.savefig(venn_path + "/venn_diagram.png")
    plt.clf()

def auc_to_binary(value, q1, q3):
    if not(value >= q3 or value <= q1):
        return "remove"
    else:
        return value

def make_result_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def generate_feature_sets(drug):
    #drug_response = pd.read_excel(url + "variants_BeatAML.xlsx", sheet_name="Table S10-Drug Responses", usecols = ['inhibitor', 'lab_id', 'auc', 'counts'])
    #drug_response = drug_response[drug_response['counts'] > 300]
    #inhib = drug_response['inhibitor'].values
    #inhib = np.unique(inhib)
    #drug_response['inhibitor'] = drug_response['inhibitor'].apply(lambda x: x.split(' ')[0])
    #m_dataset = pd.read_excel(url + "variants_BeatAML.xlsx", sheet_name="Table S9-Gene Counts CPM", dtype = 'float64', converters = {'Gene': str, 'Symbol': str})
    #k_dataset = pd.read_csv(limma_path + "read_count_matrix.txt", sep="	")
    #k_cols = k_dataset.columns.str.replace("X","-", regex=True)
    #dr = set(drug_response['lab_id'])
    #m_cols = set(m_dataset.columns)
    #k_cols = set(k_cols)
    #print("Dr")
    #print(dr)
    #print("m_cols")
    #print(m_cols)
    #print("k_cols")
    #print(k_cols)
    #print(inhib)
    #print(m_dataset)
    #total_count = len(list(set(drug_response['lab_id']) & set(m_dataset.columns) & set(k_cols)))
    #print(total_count)

    #make_result_dir(results_path)
    #inhib_results = []
    feature_size = 30
    #for i in inhib:
    dea = pd.read_csv(limma_path + drug_name + "_results.txt", sep="\t")
        #log = dea['logFC'].sort_values(ascending = True)
    log = dea['logFC']
    q1 = log.quantile(.25)
    q3 = log.quantile(.75)
    log = log.apply(lambda x: auc_to_binary(x, q1, q3))
    log = log.drop(log.index[log == 'remove'])
    #log = log.abs().sort_values(ascending = True)
    log = log.abs()
    index = np.argpartition(log, -(feature_size))[-(feature_size):]
    print(index)
    slice = log.iloc[index]
    #with open(results_path + i.split(" ")[0] + "_genes_selected.tsv", 'w') as file:
    with open(results_path + drug + "_genes_selected.tsv", 'w') as file:    
        for row in slice.index:
            s = "".join(map(str, row))
            file.write(s+'\n')
        #slice.to_csv(results_path + i + "_selected_features.txt")
        #np.savetxt(results_path + i + "_selected_features.txt", slice.values(), delimiter=",")

        #log = log[log.isin([0, 1])]
        #print(slice)
        #labels = labels.drop('auc', axis = 1)
        #print(q1)
        #print(q3)

        #pvalue = dea['P.Value'].sort_values(ascending = True)
        #print(log)
        #print(pvalue)
        #inhib_results.append(dea)
    #print(inhib_results[0])
    #print(m_dataset)
    #print(k_dataset)

def check_overlap():

    #drug_response = pd.read_excel(url + "variants_BeatAML.xlsx", sheet_name="Table S10-Drug Responses", usecols = ['inhibitor', 'counts'])
    #drug_response = drug_response[drug_response['counts'] > 300]
    #inhibitors = drug_response['inhibitor'].values

    inhibitors = ["AZD1480", "CYT387", "Elesclomol", "Flavopiridol", "Rapamycin", "Sorafenib", "STO609", "YM-155", "DBZ"]

    for fold in range(k):
        for inhib in inhibitors:
            limma_subset = pd.read_csv(results_path + inhib + "_genes_selected.tsv", header = None, names = ['DGE'])
            shap_subset = pd.read_csv(shap_path_1 + inhib + shap_path_2 + str(fold) + "/genes_selected.tsv", header = None, names = ['SHAP'])
            print("LIMMA")
            print(limma_subset)
            print("SHAP")
            print(shap_subset)
            plot_venn_diagram(results_path, inhib, limma_subset, shap_subset, fold)

generate_feature_sets(drug_name)
#check_overlap()
