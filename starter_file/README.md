# Predicting diamond price 

In this project we use a dataset which contains the features of round polished diamonds such as the 4C's (cut, clarity, colour and carat) to predict the price of the diamond. First, AutoML is used to find an optimal regression model. This is compared to a Histogram Boosted Gradient Model whose hyperparameters have been optimised using Hyperdrive. The best model from both experiments is registered in the workspace. Both models are evaluated on test data and the best one is deployed as a webservice.


## Project Set Up and Installation
In order to run this project version 1.47 of the Azure ML SDK must be used. 

## Dataset

### Overview
The dataset that will be used in this experiment is from Kaggle. It is a tablular dataset that contains 10 features related to polished diamonds such as the cut, carat weight and colour. The aim of the data is to use these features to predict the price of the polished diamond. Most of the features are numeric but the cut, clarity and colour are categorical variables which are ordinal in nature. 

The dataset can be downloaded from:
www.kaggle.co./datasets/nancyalaswad90/diamond-prices

### Task
The features from the dataset are used to predict the price for round polished diamonds in dollars.

### Access
The dataset was uploaded and registered to the Azure Machine Learning Studio. We access the dataset using its key within the Jupyter notebook and the training script.

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
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
