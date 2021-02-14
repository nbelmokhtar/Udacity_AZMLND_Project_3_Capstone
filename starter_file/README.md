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

An Azure Auto ML was performed to predict if the patient deceased during the follow-up period (DEATH_EVENT : boolean), based on the 12 clinical features. We didn't explicitly specified either a validation_data or n_cross_validation parameter, automated ML applies default techniques depending on the number of rows provided in the single training_data=dataset. Dataset is less than 1,000 rows, 10 folds are used. The Auto ML settings and configuration we used for this experiment are shown below :

```python
# AutoML settings
automl_settings = {
    "experiment_timeout_minutes": 30,
    "max_concurrent_iterations": 4,
    "primary_metric" : 'accuracy',
}

# AutoML config
automl_config = AutoMLConfig(compute_target=compute_target,
                             task = "classification",
                             training_data=dataset,
                             label_column_name="DEATH_EVENT",   
                             path = project_folder,
                             enable_early_stopping= True,
                             debug_log = "automl_errors.log",
                             **automl_settings
)
```

### Results

AutoML tries different models and algorithms during the automation and tuning process within a short period of time. Below are thescreenshots of the `RunDetails` widget.  

![RunDetails 01](https://github.com/nbelmokhtar/Udacity_AZMLND_Project_3_Capstone/blob/master/starter_file/screenshots/automl/001.PNG)
![RunDetails 02](https://github.com/nbelmokhtar/Udacity_AZMLND_Project_3_Capstone/blob/master/starter_file/screenshots/automl/002.PNG)
![RunDetails 03](https://github.com/nbelmokhtar/Udacity_AZMLND_Project_3_Capstone/blob/master/starter_file/screenshots/automl/003.PNG)

The best performing model was `Voting Ensemble` with an accuracy of 87.62%.

![Best Model](https://github.com/nbelmokhtar/Udacity_AZMLND_Project_3_Capstone/blob/master/starter_file/screenshots/automl/008.PNG)

The parameters generated by the AutoML run are :

![AutoML Parameters 01](https://github.com/nbelmokhtar/Udacity_AZMLND_Project_3_Capstone/blob/master/starter_file/screenshots/automl/004.PNG)
![AutoML Parameters 02](https://github.com/nbelmokhtar/Udacity_AZMLND_Project_3_Capstone/blob/master/starter_file/screenshots/automl/005.PNG)

We would like to tune more config parameters; increasing experiment timeout minutes will enable us to test more models and thus improving the performance.

## Hyperparameter Tuning

In this experiment, we used Logistic Regression from SciKit Learn because the prediction outcome in this case is binary. 
The goal consists of optimizing (tuning) the hyperparameters of a logistic regression binary classification model using HyperDrive. We start by setting up a training script 'train.py' were we create a dataset, train and evaluate a logistic regression model from Scikit-learn. Then, we used Jupyter Notebook and Azure ML Python SDK to get the workspace and experiment objects running, and build the training pipeline - from creating a computer cluster, to HyperDrive, to runnning the 'train.py'.

We used RandomParameterSampling method over the hyperparameter search space to randomly select values for C (choice among discrete values 0.01, 1.0, 3.0) and max_iter (choice among discrete values 50, 150, 200) hyperparameters. We used a limited number of parameters to make the experiment complete faster. Random sampling supports both discrete and continuous hyperparameters and allows us to refine the search space to improve results.

We also used BanditPolicy which defines an early termination policy based on slack_factor=0.1 and evaluation_interval=2. The slack_factor is the ratio used to calculate the allowed distance from the best performing experiment run. The evaluation_interval is the frequency for applying the policy.

```python
# Create an early termination policy. We are using Random Parameter Sampling.
early_termination_policy = BanditPolicy(evaluation_interval=2, slack_factor=0.1)

# Create the different params that you will be using during training
param_sampling = RandomParameterSampling({
        '--C': choice(0.01, 1.0, 3.0),
        '--max_iter': choice(50, 100, 150)
    }
)

script_dir = "./training"
if "training" not in os.listdir():
    os.mkdir(script_dir)
    
shutil.copy('train.py', script_dir)

# Create a SKLearn estimator for use with train.py
estimator = SKLearn(source_directory=script_dir, entry_script='train.py', compute_target=compute_target)

# Create a HyperDriveConfig using the estimator, hyperparameter sampler, and policy.
hyperdrive_run_config = HyperDriveConfig(estimator=estimator, 
                             hyperparameter_sampling=param_sampling,
                             policy=early_termination_policy,
                             primary_metric_name='Accuracy', 
                             primary_metric_goal=PrimaryMetricGoal.MAXIMIZE, 
                             max_total_runs=24,
                             max_concurrent_runs=4)
```

### Results

Here are screenshots of the `RunDetails` widget as well as a screenshot.

![HyperDrive RunDetails 01](https://github.com/nbelmokhtar/Udacity_AZMLND_Project_3_Capstone/blob/master/starter_file/screenshots/hdr/009.PNG)

The Best Run reached an accuracy of 78.33% with C = 3 and max_iter = 150.

![Best HyperDrive 01](https://github.com/nbelmokhtar/Udacity_AZMLND_Project_3_Capstone/blob/master/starter_file/screenshots/hdr/001.PNG)
![Best HyperDrive 02](https://github.com/nbelmokhtar/Udacity_AZMLND_Project_3_Capstone/blob/master/starter_file/screenshots/hdr/002.PNG)
![Best HyperDrive 03](https://github.com/nbelmokhtar/Udacity_AZMLND_Project_3_Capstone/blob/master/starter_file/screenshots/hdr/003.PNG)
![Best HyperDrive 04](https://github.com/nbelmokhtar/Udacity_AZMLND_Project_3_Capstone/blob/master/starter_file/screenshots/hdr/004.PNG)
![Best HyperDrive 05](https://github.com/nbelmokhtar/Udacity_AZMLND_Project_3_Capstone/blob/master/starter_file/screenshots/hdr/005.PNG)

The parameters generated by the HyperDrive run are :

![HyperDrive 01](https://github.com/nbelmokhtar/Udacity_AZMLND_Project_3_Capstone/blob/master/starter_file/screenshots/hdr/006.PNG)

We can also tune other hyperparameters used in Sklearn Logistic Regression in order to achieve better results in the future. Using different parameter sampling techniques and tuning the arguments of the BanditPolicy can also prove fruitful.

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
