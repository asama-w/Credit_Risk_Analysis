# Supervised Machine Learning: Credit Risk Analysis

## Analysis Overview
### Purpose of the Analysis

Using use a Python machine learning library, Scikit-learn, to train and apply six different machine learning models to analyze the LendingClub's credit card dataset and predict the credit risk of their customer. Each model employs different technique which will be evaluated whether it is suitable for the  credit risk prediction. 

### Resources
+ **Programming Language:** Python (Machine Learning)
+ **Libraries:** `scikit-learn`, `imbalanced-learn`, `pandas`
+ **Software platform:** Jupyter Notebook
+ **Raw Dataset:** Credit card dataset from LendingClub

## Results
+ **Resampling with Logistic regression Models** (Deliverable 1 and 2)
  + **Script:** [credit_risk_resampling.ipynb](https://github.com/asama-w/Credit_Risk_Analysis/blob/main/credit_risk_resampling.ipynb)
  + The first 4 algorithms will use the resampled approach and apply with the Logistic regression model.
+ **Ensemble Models** (Deliverable 3)
  + **Script:** [credit_risk_ensemble.ipynb](https://github.com/asama-w/Credit_Risk_Analysis/blob/main/credit_risk_ensemble.ipynb)
  + The last 2 algorithms will be applied with Ensemble model
### Machine Learning Models:
There are 6 machine learning models on which the dataset will be applied, using two different classifier models:
|Classifiers|Machine Learning Model|
|-----|-----|
|Logistic Regression|**Oversampling:** <br /> 1. `RandomOverSampler` <br /> 2. `SMOTE` <br /><br />  **Undersampling:** <br />3. `ClusterCentroids`<br /><br />  **Combination (Over-and-Under) Sampling:** <br /> 4. `SMOTEENN`|
|Ensemble|5. `BalancedRandomForestClassifier`<br />6. `EasyEnsembleClassifier` |


### 1. RandomOverSampler Model
+ **Balanced Accuracy Score:** 65.7%

<img src= https://github.com/asama-w/Credit_Risk_Analysis/blob/main/Images/1.png width="80%" height="80%">

### 2. SMOTE Model
+ **Balanced Accuracy Score:** 66.2%
 
<img src= https://github.com/asama-w/Credit_Risk_Analysis/blob/main/Images/2.png width="80%" height="80%">

### 3. ClusterCentroids Model
+ **Balanced Accuracy Score:** 55.5%

<img src= https://github.com/asama-w/Credit_Risk_Analysis/blob/main/Images/3.png width="80%" height="80%">

### 4. SMOTEENN Model
+ **Balanced Accuracy Score:** 64.9%

<img src= https://github.com/asama-w/Credit_Risk_Analysis/blob/main/Images/4.png width="80%" height="80%">

### 5. BalancedRandomForestClassifier Model
+ **Balanced Accuracy Score:** 78.8%

<img src= https://github.com/asama-w/Credit_Risk_Analysis/blob/main/Images/5.png width="80%" height="80%">

### 6. EasyEnsembleClassifier
+ **Balanced Accuracy Score:** 93.2%

<img src= https://github.com/asama-w/Credit_Risk_Analysis/blob/main/Images/6.png width="80%" height="80%">
