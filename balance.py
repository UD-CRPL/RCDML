import pandas as pd
import data_preprocess as dp
import sys

import numpy
import imblearn.under_sampling as us
from collections import Counter


def auc_to_binary(value, q1, q3):
    if value >= q3:
        return 1
    elif value <= q1:
        return 0
    else:
        return -1

## Loads the corresponding high responder/low responder labels for "drug_name" from the BeatAML Project
def load_labels_beatAML(url, drug_name):
    labels = pd.read_excel(url + "variants_BeatAML.xlsx", sheet_name="Table S10-Drug Responses", usecols = ['inhibitor', 'lab_id', 'auc', 'counts'], engine="openpyxl")
  #  print(labels.sort_values(by=['counts']))
  
    # Gets rid of any drugs that was tested on less than 300 samples
    labels = labels[labels['counts'] > 300]
    labels = labels.drop('counts', axis = 1)
    # Modifies the drug names so that only the first name is used (Gets rid of everything that's inside the parenthesis)
    # This makes it easier for performing operations based on drug names and saving results
    labels['inhibitor'] = labels['inhibitor'].apply(lambda x: x.split(' ')[0])
    # Checks if "drug_name" exists in the dataset
    drug_count = {}
    counter = 0
    for drug_name in set(labels['inhibitor'].tolist()):
        counter = counter + 1
        print(drug_name + ": " + str(counter))
      #  if labels['inhibitor'].str.contains(drug_name).any():
            # Selects the "drug_name" drug data
        temp = labels[labels['inhibitor'] == drug_name]
        temp = temp[['lab_id', 'auc']]
            # Calculates the 1st and 3rd quantile of the AUC distribution for "drug_name"
        q1 = temp['auc'].quantile(.25)
        q3 = temp['auc'].quantile(.75)
            # Assigns classification group to each sample:
            # If the auc score <= q1, then the sample is classified as a "low responder" or "0"
            # if auc score >= q3, then the sample is classified as a "high responder" or "1"
            # anything else is classified as -1 (which gets removed later)
        temp['GROUP'] = temp['auc'].apply(lambda x: auc_to_binary(x, q1, q3))
        temp = temp.drop('auc', axis = 1)
            # Filters out any samples that fell inside the 1st and 3rd Quantile (Anything classified as -1)
        temp = temp[temp['GROUP'].isin([0, 1])]
        temp = temp.rename(columns = {'lab_id':'SID'})
     #   print(temp.shape)
        drug_count[drug_name] = temp.shape[0]
   # else:
   #     sys.exit("ERROR beatAML Project: Labels requested not available. List of available labels are ['UNC2025A', 'original']")
    return labels, drug_count


url = "/Users/mf0082/Documents/Nemours/AML/target/"
#drug_name = "all"
#labels, drug_count = load_labels_beatAML(url, drug_name)

#sorted_age = sorted(drug_count.items(), key = lambda kv: kv[1])

dataset, samples = dp.load_dataset_target(url, "cpm")
labels = dp.load_labels_target(url)


def balance_dataset(x, y, mode):

    n_jobs = -1

 #   print('Initial dataset shape %s' % sorted(Counter(y['GROUP'].items())))
    print("PERFORMING " + mode)
    
    if mode == "CNN":
        method = us.CondensedNearestNeighbour(random_state=42, sampling_strategy = 'majority')
        # n_seeds_Sint, default=1
        # Number of samples to extract in order to build the set S.
    elif mode == "ENN":
        method = us.EditedNearestNeighbours(n_neighbors = 3, sampling_strategy = 'majority')
    elif mode == "RENN":
        method = us.RepeatedEditedNearestNeighbours(n_neighbors = 3, sampling_strategy = 'majority')
    elif mode == "ALLKNN":
        method = us.AllKNN(n_neighbors = 2, sampling_strategy = 'majority')
    elif mode == "IHT":
        method = us.InstanceHardnessThreshold(random_state=42, sampling_strategy = 'majority')
    elif mode == "NM":
        method = us.NearMiss(version = 3, sampling_strategy = 'majority')
    elif mode == "NCR":
        method = us.NeighbourhoodCleaningRule(n_neighbors = 3, sampling_strategy = 'majority')
    elif mode == "OSS":
        method = us.OneSidedSelection(random_state=42, n_neighbors = 3, sampling_strategy = 'majority')
    elif mode == "random":
         method = us.RandomUnderSampler(random_state=42, sampling_strategy = 'majority')
    elif mode == "Tomek":
        method = us.TomekLinks(sampling_strategy = 'majority')
    else: 
        sys.exit("INCORRECT MODE")

  #  print(method.sample_indices_)
   # print(method.fit(x.T, y.set_index("SID").values).n_features_in_)
    #counter = range(0, len(y.set_index("SID").values))
   # print(x.T)
 #   dataset, labels = method.fit_resample(x.T, y.set_index("SID").values)
    dataset, labels = method.fit_resample(x.T, y)
    print(mode + ' Resampled dataset shape %s' % sorted(Counter(labels).items()))
    dataset = x.T.iloc[method.sample_indices_]
    labels = y.iloc[method.sample_indices_]
   # print(dataset.shape)
   # print(labels.shape)
    samples = dataset.T.columns
    

    return dataset.T, labels, samples

def unbalance_dataset(x, y, strategy):
   # print(x)
   # print(y)
    from imblearn.datasets import make_imbalance
    from sklearn.datasets import load_iris
    #data = load_iris()
    #X, y = data.data, data.target
   # print(y)
  #  counter = range(0, len(y.set_index("SID").values))
    counter = range(0, len(y.values))
  #  dataset, labels = make_imbalance(X = numpy.array(counter).reshape(-1, 1), y = y.set_index("SID")['GROUP'], sampling_strategy = strategy, random_state = 42)
    dataset, labels = make_imbalance(X = numpy.array(counter).reshape(-1, 1), y = y, sampling_strategy = strategy, random_state = 42)
   # dataset, labels = make_imbalance(X = X, y = y, sampling_strategy = strategy, random_state=42)
    print('Imbalanced Resampled dataset shape %s' % sorted(Counter(labels).items()))
 #   dataset = x.T.iloc[method.sample_indices_]
 #   labels = y.iloc[method.sample_indices_]
    index = [item for sublist in dataset for item in sublist]
    dataset = x.iloc[:,index]
    labels = y.iloc[index]
  #  print(dataset)
  #  print(dataset.shape)
  #  print(labels.shape)

    return dataset, labels, dataset.columns
#print(dataset)
#print(labels)

#modes = ["CNN", "ENN", "RENN", "ALLKNN", "IHT", "NM", "NCR", "OSS", "random", "Tomek"]

def ratio_multiplier(y):
    from collections import Counter
   # temp = y.set_index("SID")['GROUP']
    multiplier = {0: 0.70}
    #multiplier = {0: 0.70, 1: 1.00}
    target_stats = Counter(y)
    for key, value in target_stats.items():
        if key in multiplier:
            target_stats[key] = int(value * multiplier[key])
    return target_stats

modes = ["random"]

#for mode in modes:
 #   temp, temp2, temp3 = balance_dataset(dataset, labels, mode)

#unbalanced_x, unbalanced_y, samples = unbalance_dataset(temp, temp2, {0: 10, 1: 20})
#print(unbalanced_x)
#print(unbalanced_y)
