# Capstone - Azure Machine Learning Engineer

This project is built as a part of the Machine Learning Engineer for Microsoft Azure Nanodegree Program on Udacity. In this project, we created two models: one using Automated ML (AutoML) and a Scikit-learn Logistic Regression model whose hyperparameters are tuned using HyperDrive. Then, we compared the performance of both the models and deployed the best performing model as a web service.

The diagram below shows the main steps followed to complete this project.

![Capstone Diagram](https://github.com/nbelmokhtar/Udacity_AZMLND_Project_3_Capstone/blob/master/starter_file/screenshots/capstone-diagram.png)

## Project Set Up and Installation

The starter files that we need to run this project are :

- `hyperparameter_tuning.ipynb` : Jupyter Notebook file to train a model and perform hyperparameter tuning using HyperDrive.
- `automl.ipynb` : Jupyter Notebook file to train a model using Automated ML. 
- `train.py` : Script used in Hyperdrive.
- `score.py` : Script used to deploy the model.
- `heart_failure_clinical_records_dataset.csv` : The dataset.

## Dataset

### Overview

We used the [Heart Failure Prediction dataset](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data) from `Kaggle` to build a classification model.

Dataset from Davide Chicco, Giuseppe Jurman: Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. BMC Medical Informatics and Decision Making 20, 16 (2020).

### Task

Heart failure is a common event caused by Cardiovascular diseases (CVDs) and this dataset contains 12 features :

01- `age` : Age of the patient (years)<br/> 
02- `anaemia` : Decrease of red blood cells or hemoglobin (boolean)<br/> 
03- `creatinine_phosphokinase` : Level of the CPK enzyme in the blood (mcg/L)<br/> 
04- `diabetes` : If the patient has diabetes (boolean)<br/> 
05- `ejection_fraction` : Percentage of blood leaving the heart at each contraction (percentage)<br/> 
06- `high_blood_pressure` : If the patient has hypertension (boolean)<br/> 
07- `platelets` : Platelets in the blood (kiloplatelets/mL)<br/> 
08- `serum_creatinine` : Level of serum creatinine in the blood (mg/dL)<br/> 
09- `serum_sodium` :Level of serum sodium in the blood (mEq/L)<br/> 
10- `sex` : Woman or man (binary)<br/> 
11- `smoking` : If the patient smokes or not (boolean)<br/> 
12- `time` : Follow-up period (days)<br/>

These 12 feaytures are used to predict mortality by heart failure indicated by a boolean value : `DEATH_EVENT`.
 
### Access

We used 2 ways to access the data in the workspace :

- In AutoML : we used `Dataset.get_by_name()` function to download dataset as a csv file and register it in the workspace.

- For Hyperdrive : `TabularDatasetFactory.from_delimited_files()` in the `train.py` script to create a TabularDataset to represent tabular data in delimited [CSV file](https://raw.githubusercontent.com/nbelmokhtar/Udacity_AZMLND_Project_3_Capstone/master/starter_file/heart_failure_clinical_records_dataset.csv).

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording

Here's a [screen recording](https://www.dropbox.com/s/yl9b4pqcduo3je3/nb-udacity-azmlnd-project-3.mp4?dl=0) that demonstrate :

- A working model.
- Demo of the deployed  model.
- Demo of a sample request sent to the endpoint and its response.

## Standout Suggestions

Here are some suggestions for a whole bunch of additional things we can do with this project.

- Convert the model to ONNX format.
- Deploy the model to the Edge using Azure IoT Edge.
- Enable logging in the deployed web app.

For AutoML :

- Increase "experiment_timeout_minutes" in order to achieve better results and also opt for deep learning capability.

For HyperDrive :

- Use Deep Learning models like ANNs and CNNs through Keras, TensorFlow, or PyTorch.

For Dataset :

- Perform feature engineering on the dataset.
- Add more data.
