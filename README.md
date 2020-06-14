# M5 Forecasting (Accuracy) Challenge - Code Listings

### Team MuchLearningSuchWow

This git repository contains all of the code we wrote for the M5 Forecasting (Accuracy) challenge, for the Machine Learning in Practice course. The "LSTM" folder contains code for the neural network part of our experiment, and the "LightGBM" folder contains the code for the LightGBM part of our experiment.
The "Other" folder contains notebooks for WRMSSE evaluation and prediction analysis. 
This repository does not contain any data; data can be downloaded from the [kaggle competition](https://www.kaggle.com/c/m5-forecasting-accuracy/overview).

#### LSTM Instructions

All notebooks containing our code are listed directly in the LSTM folder. In order to test different model additions, run:
- LSTM_Downcasting.ipynb
- LSTM_Preprocessing.ipynb (with the correct booleans set at the top of the notebook)
- LSTM_Crossvalidation.ipynb (with the correct booleans set at the top of the notebook)

To train a final model and create predictions, run:
- LSTM_Downcasting.ipynb
- LSTM_Preprocessing.ipynb (with the correct booleans set at the top of the notebook)
- LSTM_Training.ipynb (with the correct booleans set at the top of the notebook)
- LSTM_Testing.ipynb (with the correct booleans set at the top of the notebook)



#### LightGBM Instructions

All source files are located in `LightGBM/m5_forecasting`. 
- `__main__.py`: The main entrypoint of the package.
- `loader.py`: Data loading utility functions
- `models.py`: Neural regression and LightGBM models
- `preprocessing.py`: All available preprocessing steps (which are instantiated based on the preprocessing .json files)
- `utils.py`: Miscellaneous utilities

`config/preprocessing` contains various JSON files which define different preprocessing procedures used to generate input data for training/inference. 

`notebooks` and `kaggle_notebooks` contain various Jupyter notebooks, which are not directly relevant for the main source code. 

`tests` contains a few unittests (also obviously not necessary to understand the main code). 

###### How to reproduce
1. `pip install -e .`
2. `python -m m5_forecasting preprocess config/preprocessing/preprocess_04.json --output data/processed.h5`
3. `python -m m5_forecasting train data/processed.h5 -output model_new.txt -model lightgbm`
4. `python -m m5_forecasting predict config/preprocessing/preprocess_04_eval.json lightgbm model.txt preds.csv --magic`