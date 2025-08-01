# DMCRL: Deep Multi-View Contrastive Representation Learning for Drug–Target Affinity Prediction

## Project Structure

`Args.py`: Argument parsing and default hyperparameter setting.

`Data.py`: Handles data loading, SMILES/sequence parsing, graph construction (nodes, edges), and dataset split logic.

`Models.py`: Contains all core model components including:

- Drug encoder (GCN),
- Protein encoder (CNN with residual blocks),
- Affinity graph encoder (GNN),
- SimCLR regularization,
- Final predictor.

`TrainEvaluate.py`: Training, validation, early stopping, and testing pipeline.

`Experiments.py`: Entry point. Controls different modes (optuna search or training).

`utils.py`: Metrics and evaluation (CI, RMSE, R², etc.).

## Dataset

All datasets used in this research are openly available from these sources:

- Davis and KIBA(regression): https://github.com/hkmztrk/DeepDTA/tree/master/data  
- Human and *C.elegans*(classification): https://github.com/masashitsubaki/CPI_prediction  
- ToxCast(regression): https://github.com/simonfqy/PADME  
  Or you can download the datasets at https://zenodo.org/records/13235781

## Requirements  

torch==2.5.0  
NumPy==1.26.4  
torch_geometric==2.6.1  
SciPy==1.14.1  
scikit-learn==1.5.2  
pandas==2.2.2  
tqdm==4.66.5  
networkx==3.3   
rdkit==2024.03.5  
Optuna==4.4.0 

## Step-by-step running:  

### 1. Run Optuna hyperparameter search

`python Experiments.py --dataName kiba --expName optuna --n_trials 20`  

### 2. **Run training and testing**

`python Experiments.py --dataName davis --expName runModel`  
