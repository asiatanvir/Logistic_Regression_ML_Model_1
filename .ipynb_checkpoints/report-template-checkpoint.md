# Module 12 Report Template

## Overview of the Analysis

This jupyter notebook is an application of the scikit-learn machine learning library that builds a loan classification model to identify the creditworthiness of borrowers. The dataset of historical lending activity from a peer-to-peer lending services company consisted of 77536 records. A number of features including loan size, interest rate, debt burden, debt to income ratio determined the loan health.The data set is highly imbalanced, our targeted column (y)consists of 75036 healthy loans(97%) while only 2500 records for high risk loans(3%). To build this model, data has been splitted to default ratio of 75/25 % of train test split as part of pre-processing. 

Since our targeted column would result into a yes or no answer therefore logistic regression model has been used for classification.Logistic regression is a famous classification technique when dealing with the binary data. This technique uses a logistic function to model the dependent variable. The dependent variable is dichotomous in nature, i.e. there could only be two possible classes (eg.: either the loan is helathy or risky)(1). 

In the first part of analysis, logistic regression model is initiated based upon the original data. After training the model(X_train), predictions on the testing data labels by using the testing feature data (X-test) generated 95% balanced accuracy score of the model. The model performed very well upon the healthy loans, however it is week for defaulting loans which could be due to imbalanced training and testing data. Please note that the original data is biased towards healthy loans since they makeup a large proportion of the data.

To cater the skeweness of data, second part of analysis resampled the data with "RandomOverSampler". Random oversampling is a basic sampling method used for increasing the number of the minority class. Data points from the minor class are randomly selected and duplicated exactly in this method resulting an increase, the number of minority samples to create a balance between both classes(2). After resampling,  X(features) and y(target) transformed to 56271 records for healthy loans and 56271 records for risky loans. This resampled data increased the balanced accuracy socre of our logistic regression model from 95.20% to 99.37%.

A comparison of classification report for original data and resampled data is as below.

|classification report (original Data)                    | classification report (resampled data)                       |
| -----------------------------------                     | ----------------------------------- |
| ![image 1](classification_original.png.png)             | ![image_2](classification_resampled.png) |




## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1 (original data):
  * Balanced Accuracy Score: 0.9520
  * Precision for Healthy loans: 1.00
  * Recall for Healthy loans :   0.99
  * F1 for risky loans:  1.00
  * Precision for riskyloans: 0.85
  * Recall for risky loans :  0.91
  * F1 for risky loans:       0.88

* Machine Learning Model 2 (Resample data):
  * Balanced Accuracy Score: 0.9937
  * Precision for Healthy loans: 1.00
  * Recall for Healthy loans : 0.99
  * F1 for risky loans:    1.00
  * Precision for riskyloans: 0.84
  * Recall for risky loans :  0.99
  * F1 for risky loans:      0.91


## Summary

Machine learning model with original data consisting of healthy loans as majority predcited very well for the healthy loans. It's precision, recall and f1 were very high reflecting the effectiveness of model for classification and prediction of healthy loans. However, the model is week for risky loans as recall and precision are quite less in comparison of healthy loans. Since data is imbalanced and false prediction of a loan could cost a significant loss to bank/ lending company if the client actually defaults in future.

The second model with resampled data improved the results significantly for risky loans. The over all accuracy of model increased to 99.37%.  Revised model's results did not change for healthy loans however recall for risky loans increased impressively from 91% to 99%. This makes it more attractive for prediction of risky loans. Since prediction of non defaulting clients (False negatives) could be prevented by maximizing the recall.  Low precision raises a flag to the effectiveness of model but prediction of a client as False positive (predicted a borrower as to default whereas actually they are not defaulting) would not lead the company to direct loss of default.Therefore in this case emphasizing upon recall rather than precision or f1 is more significant.


Based upon the comparative performance anlysis logistic regression model with resampling performed fairly decent. Random sampling has prevented from loss of any information for healthy loans but it  creates  noise for duplicated minority data. The model trained on such balanced data may not perform well with the real world unseen imbalanced data. Therefore I would suggest insead of using random sampling the other techniques such as SMOTE and SMOTEEN should be evaluated. Additionally although logistic regression is easy to trained for binary data but I would recomend to utilize Random Forest to identify high quality peer to peer borrowers.

Refernces:

1.https://medium.com/geekculture/mastering-loan-default-prediction-tackling-imbalanced-datasets-for-effective-risk-assessment-8e8dfb2084d0#:~:text=Out%20of%20all%20the%20clients%20who%20are%20predicted%20to%20default,priority%20than%20Precision%20and%20Accuracy.
2.https://towardsdatascience.com/oversampling-and-undersampling-5e2bbaf56dcf