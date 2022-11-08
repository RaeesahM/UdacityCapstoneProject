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
The AutoML task is set to regression as we want to predict a numerical value (diamond price) from a set of features. We will use the compute cluster that already exists in the workspace to train the model. The experiment is set to time out after 30 minutes so that we don't consume an indefinite amount of resources. The primary metric that will be optimised is normalized_root_mean_squared_error as we want to minimise the average squared error. As the dataset is not very large, 5 cross-validations are used. Featurisation is enabled as no data cleaning was done after reading in the data and inputting it to the AutoML model. Early stopping is also enabled.

### Results
A total of 23 models were trained by the AutoML model. The best model was a Voting Ensemble. This makes sense as it combines the results from many models. Looking at the feature importance (using Explain Model) we see that the carat,y cut and clarity were the most important features. Model improvement may be improved by using feature selection before model training. In particular some cleaning could be done to remove the "Column2" feature as this is just the index of the sample in the dataset and is not correlated to the diamond price. 

The RunDetails widget shows the completed runs.
![alt text](AutoMLRunDetailsScreenshot.png)

The screenshot below shows the best model trained with its parameters:
![alt text](AutoMLBestRunModel.png)


## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

Gradient boosting is an ensemble of decision trees algorithms. It is a popular techniques for tabular regression predictive modeling problems given that it performs well across a wide range of datasets in practice. A major problem of gradient boosting is that it is slow to train the model. This is particularly a problem when using the model on large datasets with tens of thousands of examples as is the case for the diamond dataset. This training can be accelerated by binning the continuous variables so that the input variables can be reduced to fewer unique values. This is then referred to as histogram gradient bootsing and is the model employed for the diamond dataset hyperdrive experiment.

In the experiment, two hyperparameters are tuned the learning rate and the maximum tree depth. The learning rate controls determines the impact of each tree on the final outcome. GBM works by starting with an initial estimate which is updated using the output of each tree. The learning parameter controls the magnitude of this change in the estimates. Lower values are generally preferred as they make the model robust to the specific characteristics of tree and thus allowing it to generalize well. Lower values would require higher number of trees to model all the relations and will be computationally expensive. In the experiment the learning rate . The maximum tree depth is the maximum depth of a tree. It is used to control over-fitting as higher depth will allow model to learn relations very specific to a particular sample. In the experiment this is a choice of values between 8 and 30. We saw that higher values for this resulted in a smaller mean squared error but when evaluating the best model on a test set the model did not perform as well.




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
