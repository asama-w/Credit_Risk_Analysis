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
There are 6 machine learning models on which the dataset will be applied to predict the loan status (high risk/low risk), using two different classifier models:
|Classifiers|Machine Learning Model|
|-----|-----|
|Logistic Regression|*Oversampling:* <br /> 1. `RandomOverSampler` <br /> 2. `SMOTE` <br /><br />  *Undersampling:* <br />3. `ClusterCentroids`<br /><br />  *Combination (Over-and-Under) Sampling:* <br /> 4. `SMOTEENN`|
|Ensemble|5. `BalancedRandomForestClassifier`<br />6. `EasyEnsembleClassifier` |

### Target Variables Balance before resampling:
+ The target of this analysis is the "**Loan Status**", whether the customer is at **high risk** or **low risk**.
<img src= https://github.com/asama-w/Credit_Risk_Analysis/blob/main/Images/target-variable-balance.png width="70%" height="70%">

### 1. RandomOverSampler Model (Oversampling)
+ The RandomOverSampler randomly duplicate samples of the minority class (high-risk) to match with the number of majority class (low_risk), of which in this model the trained target variable is 51,366 ('low_risk': 51366, 'high_risk': 51366).
+ **Balanced Accuracy Score: 65.7%**, meaning that the model did quite moderately at predicting the credit risk.
+ The **precision of the high_risk is extremely low, as of 0.01 (1%)** whereas the precision of low_risk is 1 (100%)
+ The **recall or the sensitivity of the high_risk is 0.71 (71%)** and 0.6 (60%) for low_risk. From recall velue, the model might have a moderately-high chance of predicting the right number of high-risk out of the actual high-risk pool. However, since the precision is very low, it is unlikely that the those who are predicted to be at high risk will actually have a high credit risk.
+ **The F1 score of high-risk class is 0.02, which is also very low**, meaning that the sensitivity and precision is imbalance. Hence, this model is not suitable for the credit risk prediction.

<img src= https://github.com/asama-w/Credit_Risk_Analysis/blob/main/Images/1.png width="80%" height="80%">

### 2. SMOTE Model (Oversampling)
+ The number of the trained data samples of minority class (high-risk) is duplicated to 51,366 ('low_risk': 51366, 'high_risk': 51366).
+ **Balanced Accuracy Score:** 66.2%**, which is slightly higher than the previous RandomOverSampler model, however, it still implies that the model performance is still in the moderate range.
+ The **precision for high-risk and low-risk is 0.01 and 1**, respectively.
+ The recall values is 0.69 for low-risk. The value dropped slightly for the high-risk to 0.63.
+ Overall, the resulted performances are quite similar to the RandomOverSampler model, the **F1 score of high-risk is still very low at 0.02.** This model is also not suitable for the credit risk prediction.
 
<img src= https://github.com/asama-w/Credit_Risk_Analysis/blob/main/Images/2.png width="80%" height="80%">

### 3. ClusterCentroids Model (Undersampling)
+ The ClusterCentroids Model reduces the number of samples in the majority class to match with the minority's, of which in this case, the number of trained majority's target variable (low-risk) is reduced to 246 (number of samples: 'high_risk': 246, 'low_risk': 246).
+ **Balanced Accuracy Score: 55.5%**, which is the lowest among the 6 models in this analysis. The model does not perform very well for predicting the credit risk.
+ The precision of high risk and low-risk is 0.01 and 1, respectively, which is similar to those of the previous two models.
+ The recall and F1 value of the low-risk dropped significantly (low-risk recall = 0.4, low-risk F1 = 0.57), if compared to the previous oversampling models, resulting from the major reduce in the number of low-risk data.
+ The F1 value of high-risk is even lower to 0.01, which means that there are more imbalances between precision and sensitivity of the high-risk.
+This model is also not suitable for the prediction.

<img src= https://github.com/asama-w/Credit_Risk_Analysis/blob/main/Images/3.png width="80%" height="80%">

### 4. SMOTEENN Model (Combination or Over-and-UnderSampling)
+ The number of the trained target variables of both the majority (low-risk) and minority class (high-risk) are resampled to balance each other out, the number of trained target samples in this model is as follow: 'high_risk': 68460, 'low_risk': 62011.
+ **Balanced Accuracy Score: 64.9%**
+ The recall value of the high-risk is 0.72 or 72%, which is the highest among the four resampling models. However, the recall of the low-risk and the average recall value are only slighly over 50%.
+ The high-risk's F1 score is still as low as 0.02. 
+ Comparing all the values including the average values, this model's performance is quite similar to the oversampling's RandomOverSampler Model, which may still not be a fit fot the credit risk application.

<img src= https://github.com/asama-w/Credit_Risk_Analysis/blob/main/Images/4.png width="80%" height="80%">

### 5. BalancedRandomForestClassifier Model
+ The number of target variables in the train group is 'low_risk': 51366, 'high_risk': 246.
+ **Balanced Accuracy Score: 78.8%** which is higher than all of the resampling models.
+ The recall values increased noticably for the low-risk to 0.87 (87%), meaning that more actual low-risk credit is correctly detected as low-risk.
+ The precision of high-risk is still low, despite an increase to 0.03.
+ The F1 score also increases to 0.06 for high-risk and 0.93 for low-risk.


<img src= https://github.com/asama-w/Credit_Risk_Analysis/blob/main/Images/5.png width="80%" height="80%">

### 6. EasyEnsembleClassifier
+ The number of target variables in the train group is 'low_risk': 51366, 'high_risk': 246.
+ **Balanced Accuracy Score: 93.2%** which is the highest of all, indicating that this model performs best among the six models on the credit risk prediction.
+ There is a jump in the *precision of high-risk to 0.09 or 9%* and a significant increase in *high-risk's F1 score to 0.16 (16%)*, despite still being a low value, it is the highest precision and F1 value for the high-risk in this analysis. This model is more balance between the precision and the recall (sensitivity).
+ **The recall value is 0.94 for low-risk, and 0.92 for high-risk**, meaning that among the actual loan status, their status are predicted correctly. 
+ **The precision, recall, F1 score values of low-risk and the average are all over 90.** Thus, this model fit best to the prediction of the credit risk.

<img src= https://github.com/asama-w/Credit_Risk_Analysis/blob/main/Images/6.png width="80%" height="80%">

## Summary
+ The precision value of low-risk for all the six models is equal to 1. In contrast, the precision value of high-risk of all models are under 0.1 with the highest of 0.09 from the EasyEnsembleClassifier model. This may results from the extreme inbalance number of the two data classes (low and high risk). The number of high-risk data is significantly small when compared to the high number of the low-risk data.
+ Among the 6 machine learning models, the **EasyEnsembleClassifier** gives the best results with the balance accuracy score of 93.2%, in other words, it is the most suitable model for the credit risk prediction if we were to pick one from this analysis. The values of precision, recall, and F1-score of the low-risk and the average are high, respectively 0.99, 0.94, 0.97, which is nearly 1, indicating a good low-risk prediction as the high F1 value shows that the precision and recall (sensitivity) is balanced. Simply put, using `EasyEnsembleClassifier` model to predict the loan status, most of the customer who has low-risk status are detected (high recall value), and among those deteced low-risk, the number of people who actually has low-risk status is also high (high precision). However, since the results for high-risk is still very much low, the model might need considerably more high-risk loan status data in order for it to better train itself to be able to predict the high-risk status effectively.
