import pandas as pd
import sys
from pathlib import Path
from sklearn.utils import resample

## Handler functions

# Handles wether to load the dataset from the BeatAML project or a different dataset
def load_dataset(url, project, normalization):
    # Loads BeatAML data
    if project.lower() == "beataml":
        dataset, samples = load_dataset_beatAML(url, normalization)
    else:
        dataset, samples = load_dataset_rnaseq(url)
    return dataset, samples

# Handles wether to load the labels from the BeatAML project or from a different dataset
def load_labels(url, project, drug_name):
    # Loads BeatAML data
    if project.lower() == "beataml":
        labels = load_labels_beatAML(url, drug_name)
    else:
        labels = load_labels_rnaseq(url)
    return labels

# Matches the samples from the dataset and labels, gets rid of any samples that are not available in both data matrices
def sample_match(dataset, labels, dataset_samples):
    labels = labels[labels['SID'].isin(dataset_samples)]
    dataset = dataset[labels['SID']]
    samples = labels['SID']
    return dataset, labels, samples

## Functions that change label notation

def category_to_binary(group):
    if group == "high":
        return 1
    elif group == "low":
        return 0
    else:
        return -1

def group_to_bool(group):
    if group == "Positive":
        return True
    elif group == "Negative":
        return False
    else:
        return -1

def bool_to_group(bool):
    if bool == True:
        return "Positive"
    elif bool == False:
        return "Negative"
    else:
        return -1

def group_to_binary(group):
    if group == "Group 1" or group == 1:
        return 0
    elif group == "Group 2" or group == 2:
        return 1
    else:
        return -1

def binary_to_group(binary):
    if binary == 0:
        return "Group 1"
    elif binary == 1:
        return "Group 2"
    else:
        return "Unknown"

def bool_to_binary(bool):
    if bool == True:
        return 0
    elif bool == False:
        return 1
    else:
        return -1

def auc_to_binary(value, q1, q3):
    if value >= q3:
        return 1
    elif value <= q1:
        return 0
    else:
        return -1

### PROJECT DATASETS

## Loads the RNA Sequence Data Matrix from the BeatAML Project
def load_dataset_beatAML(url, normalization):
    if normalization == "cpm":
        dataset = pd.read_excel(url + "variants_BeatAML.xlsx", sheet_name="Table S9-Gene Counts CPM", dtype = 'float64', converters = {'Gene': str, 'Symbol': str})
    elif normalization == "rpkm":
        dataset = pd.read_excel(url + "variants_BeatAML.xlsx", sheet_name="Table S8-Gene Counts RPKM", dtype = 'float64', converters = {'Gene': str, 'Symbol': str})
    else:
        sys.exit("ERROR BeatAML Project: Dataset requested not available. List of available datasets are ['cpm', 'rpkm']")
    # Sets the gene ID as the index for the data matrix rows. THESE GENE ROWS ARE THE FEATURES
    # Makes selection/manipulation by features easier
    dataset = dataset.set_index('Gene')
    # Drops symbol column since gene ID is already being used to track back
    dataset = dataset.drop('Symbol', axis = 1)
    # Gets the list of sample IDs from the dataset
    samples = dataset.columns
    return dataset, samples

## Loads the corresponding high responder/low responder labels for "drug_name" from the BeatAML Project
def load_labels_beatAML(url, drug_name):
    labels = pd.read_excel(url + "variants_BeatAML.xlsx", sheet_name="Table S10-Drug Responses", usecols = ['inhibitor', 'lab_id', 'auc', 'counts'])
    # Gets rid of any drugs that was tested on less than 300 samples
    labels = labels[labels['counts'] > 300]
    labels = labels.drop('counts', axis = 1)
    # Modifies the drug names so that only the first name is used (Gets rid of everything that's inside the parenthesis)
    # This makes it easier for performing operations based on drug names and saving results
    labels['inhibitor'] = labels['inhibitor'].apply(lambda x: x.split(' ')[0])
    # Checks if "drug_name" exists in the dataset
    if labels['inhibitor'].str.contains(drug_name).any():
        # Selects the "drug_name" drug data
        labels = labels[labels['inhibitor'] == drug_name]
        labels = labels[['lab_id', 'auc']]
        # Calculates the 1st and 3rd quantile of the AUC distribution for "drug_name"
        q1 = labels['auc'].quantile(.25)
        q3 = labels['auc'].quantile(.75)
        # Assigns classification group to each sample:
        # If the auc score <= q1, then the sample is classified as a "low responder" or "0"
        # if auc score >= q3, then the sample is classified as a "high responder" or "1"
        # anything else is classified as -1 (which gets removed later)
        labels['GROUP'] = labels['auc'].apply(lambda x: auc_to_binary(x, q1, q3))
        labels = labels.drop('auc', axis = 1)
        # Filters out any samples that fell inside the 1st and 3rd Quantile (Anything classified as -1)
        labels = labels[labels['GROUP'].isin([0, 1])]
        labels = labels.rename(columns = {'lab_id':'SID'})
    else:
        sys.exit("ERROR beatAML Project: Labels requested not available. List of available labels are ['UNC2025A', 'original']")
    return labels

# Creates new directory and subdirectories if given a path and the directory does not exist
# Used extensively to save results
def make_result_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

#### !!!! CAN BE MODIFIED TO FIT YOUR OWN DATASET !!!! ####

## Function to load new dataset
def load_dataset_rnaseq(url):
    dataset = pd.read_csv(url, sep='\t', index_col=0)
    samples = dataset.columns
    return dataset, samples

## Function to load new labels
def load_labels_rnaseq(url):
    labels = pd.read_csv(url, sep='\t', index_col=0)
    return labels
