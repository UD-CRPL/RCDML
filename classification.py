from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import xgboost
import numpy as np
import pandas as pd
import data_preprocess as dp

# List of hyperparameters to search for the Random Forest scikit-learn implementation
rf_parameters = {
'bootstrap': [True, False],
'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, None],
'max_features': ['auto', 'sqrt'],
'min_samples_leaf': [1, 2, 4],
'min_samples_split': [2, 5, 10],
'n_estimators': [100, 150, 200, 250, 500, 750, 1000]}

# List of hyperparameters to search for the XGBoost gradient boosting implementation
gdb_parameters = {
'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, None],
'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
'colsample_bylevel': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
'min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
'gamma': [0, 0.25, 0.5, 1.0],
'reg_lambda': [0.1, 1.0, 5.0, 10.0, 50.0, 100.0],
'n_estimators': [100, 150, 200, 250, 500, 750, 1000]}

# Classification wrapper used to select the correct classifier based on the configuration file selection
def get_model(classifier, hyper_opt):

    # Classifier: "rf"
    # Random Forest, scikit-learn
    if classifier == 'rf':
        model = RandomForestClassifier()
        # Random Search CV used for Hyperparameter optimization, sets up the operation for
        # going through the list of hyperparameters above and selects best performing model
        if hyper_opt == "random_search":
            model = RandomizedSearchCV(model, rf_parameters, n_iter=30,
                                    n_jobs=-1, verbose=0, cv=5,
                                    scoring='roc_auc', refit=True, random_state=42)
    # Classifier: "gdb"
    # Gradient Boosting, xgboost
    elif classifier == 'gdb':
        model = xgboost.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        # Random Search CV used for Hyperparameter optimization, sets up the operation for
        # going through the list of hyperparameters above and selects best performing model
        if hyper_opt == "random_search":
            model = RandomizedSearchCV(model, gdb_parameters, n_iter=30,
                                            n_jobs=-1, verbose=0, cv=5,
                                            scoring='roc_auc', refit=True, random_state=42)
    else:
        sys.exit("ERROR: Unrecognized classification technique in configuration file. Please pick one or more from these options: ['rf', 'gdb']")
    return model

# Function that tranverse the data matrix so that it matches with the sickit-learn format and converts the labels to binary format
def prepare_dataset(x, y):
    x = x.T
    y = y.apply(lambda x: dp.bool_to_binary(x))
    return x, y

# Performs the classifier training using the training dataset
def model_train(path, x, y, classifier, debug_mode, iteration, hyper_opt, best_parameters):
    # DEBUG MODE
    if debug_mode:
        # Saves input training dataset and labels
        debug_path = path + classifier + "/debug/" + str(iteration) + "/"
        dp.make_result_dir(debug_path)
        x.to_csv(debug_path + "/input_dataset.tsv", sep="\t")
        y.to_csv(debug_path + "/labels.tsv", sep="\t")

    # Selects correct model
    model = get_model(classifier, hyper_opt)
    x, y = prepare_dataset(x, y)
    if hyper_opt == "best":
        #print(best_parameters[1])
        #print(best_parameters)
        model.set_params(**best_parameters[1])
        # Transforms the dataset for correct scikit-learn format
    print("CLASSIFIER: " + classifier)
    # Trains the model
    model.fit(x, y)

    if hyper_opt == "random_search":
        print(hyper_opt)
        best_parameters = model.best_params_

    # DEBUG MODE
    if debug_mode:
        # Saves the trained model
        # Load function can be implemented to load the model back for debugging purposes
        from joblib import dump, load
        dump(model, debug_path + 'model.joblib')

    return model, best_parameters
