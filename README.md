# RCDML

Welcome to the RCDML Project!

The ***R**NA-seq **C**ount **D**rug-response **M**achine **L**earning **(RCDML)*** Workflow is a ML workflow that can be used for drug response classification of rare disease patients. Given drug response data and RNA-seq count data, the model follows the ML workflow below to classify patients as "high responder" or "low responder" for a given inhibitor.

The *RCDML* pipeline was evaluated using RNA-seq count and drug response data available for Acute Myeloid Leukemia (AML) patients and over 100 different drugs as part of the BeatAML project.

**The *RCDML* source code consist of:**

`parser.py` – Contains the code used to parse the configuration file. 

`data_preprocessing.py` – Contains the code that loads and configures the dataset, and the code for assigning responder/non-responder labels. 

`feature_selection.py` – Contains the implementations of the feature selection techniques and the feature counter generator. 

`classification.py` – Contains the implementation of the classifiers and the hyperparameter optimization technique. The hyperparameter lists used are found here.

`validation.py` – Contains the code used to create the confusion matrices, ROC curves, run inference and make predictions. 

`main.py` – Contains the framework structure. This is the script that needs to get called to run the ML pipeline. 

`parameters.cfg` – Parameter configuration for the ML pipeline run. The user can select the feature selection techniques, classifiers, and other options that will be used in the run. 

`/tools/` - Utility tools for gathering results, create feature counters, get family drug list, etc. For more information on each tool follow this wiki page link.

`/setup/` - Contains conda environment yml file and parameter configuration presets. For more information on each setup preset follow this wiki page.


For documentation on how to get started with the *RCDML* workflow visit the [wiki](https://github.com/UD-CRPL/RCDML/wiki).
