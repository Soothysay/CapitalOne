# Capital One Data Science Challenge
## Create Conda Environment from YML File
1. Run the following command to create the Conda environment:
   ```bash
   conda env create -f environment.yml
2. Look at Data Science Challenge.pptx to view a brief summary of all the methods and experiments.

## Directories
1. data/ contains all the data files used in the experiments.
2. figs/ contains all the figures that were created for replication.
3. models/ contains the codes to perform hyperparameer tuning on the ML models.
4. saved_models/ contains all the pickle files of the best ML models.
5. utils/ contains the utility functions used in the codes.

## Running Files:
1. Please run read_file_summary.ipynb to look at the answers to Question 1. Please look at the Jupyter notebook to obtain more information about the statistics and findings that are not shown in the pptx file.
2. Please run plot_things.ipynb to look at the answers to Question 2. Check this for plots not present in the powerpoint presentation.
3. Please run handle_duplicates.ipynb to look at the answers to Question 3.
4. Please run prepare_data.ipynb to create the features of the ML models used for Question 4.
5. Please run the script to perform hyperparameter tuning and save the best Decision Tree and Random Forest ML Models:
   ```bash
   python3 ml_models_tuning.py
6. Please run the script to perform hyperparameter tuning for FFN and FFN+libAUC respectively which stores the ourputs in the .out files respectively:
   ```bash
   nohup python3 modelling_FFN_optuna.py > myprogram1.out 2>&1 &
   nohup python3 libaucv1.py > myprogram.out 2>&1 &
7. Run the following command to run the evaluation of Decision Tree, Random Forest Classifier, Feed Forward Network and Feed Forward Network + libAUC:
   ```bash
   python3 all_models_roc.py
   python3 libauc_run.py
8. Run the following script to get the feature importance of Random Forest Classifier:
   ```bash
   python3 feature_importances_models.py

## Predictive Model Results

| Model | Test AUC |
|----------|----------|
|   Decision Tree  |   0.546  |
|   Random Forest  |   0.782  |
|   Feed Forward Network  |   0.784  |
|   Feed Forward + libAUC  |   0.564  |
