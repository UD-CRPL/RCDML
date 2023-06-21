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
    elif project.lower() == "target":
        dataset, samples = load_dataset_target(url)
    elif project.lower() == "pd":
        dataset, samples = load_dataset_pd(url)
    else:
        dataset, samples = load_dataset_rnaseq(url)
    return dataset, samples

# Handles wether to load the labels from the BeatAML project or from a different dataset
def load_labels(url, project, drug_name):
    # Loads BeatAML data
    if project.lower() == "beataml":
        labels = load_labels_beatAML(url, drug_name)
    elif project.lower() == "target":
        labels = load_labels_target(url)
    elif project.lower() == "pd":
        labels = load_labels_pd(url)
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
    
def vital_to_binary(value):
    if value == "Alive":
        return 1 
    else:
        return 0

### PROJECT DATASETS

## Loads the RNA Sequence Data Matrix from the BeatAML Project
def load_dataset_beatAML(url, normalization):
    if normalization == "cpm":
        #dataset = pd.read_excel(url + "variants_BeatAML.xlsx", sheet_name="Table S9-Gene Counts CPM", dtype = 'float64', converters = {'Gene': str, 'Symbol': str})
        dataset = pd.read_csv(url + "read_count_matrix.txt", dtype = 'float64', converters = {'Gene': str, 'Symbol': str}, sep="\t")
    elif normalization == "rpkm":
        dataset = pd.read_excel(url + "variants_BeatAML.xlsx", sheet_name="Table S8-Gene Counts RPKM", dtype = 'float64', converters = {'Gene': str, 'Symbol': str}, engine="openpyxl")
    else:
        sys.exit("ERROR BeatAML Project: Dataset requested not available. List of available datasets are ['cpm', 'rpkm']")
    # Sets the gene ID as the index for the data matrix rows. THESE GENE ROWS ARE THE FEATURES
    # Makes selection/manipulation by features easier
    dataset = dataset.set_index('Gene')
    # Drops symbol column since gene ID is already being used to track back
    dataset = dataset.drop('Symbol', axis = 1)
    # Gets the list of sample IDs from the dataset
    dataset.columns = [s.replace('X','-') for s in dataset.columns]
    samples = dataset.columns
    return dataset, samples

## Loads the corresponding high responder/low responder labels for "drug_name" from the BeatAML Project
def load_labels_beatAML(url, drug_name):
    labels = pd.read_excel(url + "variants_BeatAML.xlsx", sheet_name="Table S10-Drug Responses", usecols = ['inhibitor', 'lab_id', 'auc', 'counts'], engine="openpyxl")
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


def load_dataset_target(url):
    dataset = pd.read_csv(url + "genesdf.txt", sep="\t")
   # 
    dataset = dataset.drop("Symbol", axis = 1)
   # dataset.to_csv(url + "genesdf.txt", sep="\t")
    dataset = dataset.set_index('Gene')
    samples = dataset.columns
    return dataset, samples

def load_labels_target(url):
    labels = pd.read_csv(url + "target.csv")
    labels['GROUP'] = labels['GROUP'].apply(lambda x: vital_to_binary(x))
    return labels
    
def load_dataset_pd(url):
    dataset = pd.read_csv(url + "snp_matrix.csv", sep="\t")
    dataset["#CHROM-POS"] = dataset["#CHROM"].astype(str) + "-" + dataset["POS"].astype(str)
    dataset.drop("#CHROM", axis = 1)
    dataset.drop("POS", axis = 1)
    dataset = dataset.set_index('#CHROM-POS')
    samples = dataset.columns
    print(dataset)
    return dataset, samples

def load_labels_pd(url):
    labels = pd.read_csv(url + "00-PD-TreatmentCodeTable-ALL153.csv", usecols = ["SID", "GROUP"])
    labels['GROUP'] = labels['GROUP'].apply(lambda x: group_to_binary(x))
    print(labels)
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

def simulate_data(dataset, labels, simulation_size):
    
    dataset_size = len(dataset.columns)
    extra_samples_size = simulation_size - dataset_size
    
    if(extra_samples_size < 0): 
        sys.exit("Requested simulation of data that's smaller than sample size, please change the simulation size")

    extra_samples = labels.groupby("GROUP").sample(n = int(extra_samples_size / 2), random_state=1, replace = True)
    
    sampled_dataset = dataset[extra_samples["SID"]]
    
    col_names = ["simulated_sample_" + str(i) for i in range(0, extra_samples_size)]
    
    extra_samples["SID"] = col_names
    sampled_dataset.columns = col_names
    
    dataset = pd.concat([dataset, sampled_dataset], axis=1)
    labels = pd.concat([labels,  extra_samples], axis=0)

    return dataset, labels, dataset.columns

def balance_dataset(X_imbalanced, y_imbalanced):
    df = X_imbalanced.T
    df["GROUP"] = y_imbalanced.set_index("SID")
    df_majority = df[df.GROUP==1]
    df_minority = df[df.GROUP==0]
 
    # Downsample majority class
    df_majority_downsampled = resample(df_majority, 
                                 replace=False,    # sample without replacement
                                 n_samples=len(df_minority),     # to match minority class
                                 random_state=123) # reproducible results
 
    # Combine minority class with downsampled majority class
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])
 
    # Display new class counts
    print(df_downsampled.GROUP.value_counts())
    
    y_balanced = df_downsampled["GROUP"]
    y_balanced = y_balanced.reset_index()
    y_balanced = y_balanced.rename(columns={"index": "SID"})
    X_balanced = df_downsampled.drop('GROUP', axis=1).T
    
    print(y_balanced)
    return X_balanced, y_balanced, X_balanced.columns
