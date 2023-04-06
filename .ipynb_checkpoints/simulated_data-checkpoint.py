import data_preprocess as dp
import pandas as pd
import numpy as np
import sys


result_path = "/mnt/d/school_and_work/BeatAML/results/"
dataset_path = "/mnt/d/school_and_work/BeatAML/dataset/"
data, samples = dp.load_dataset_beatAML(dataset_path, "cpm")
labels = dp.load_labels_beatAML(dataset_path, 'Sorafenib')
matched_data, matched_labels, matched_samples = dp.sample_match(data, labels, samples)
simulation_size = 500

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