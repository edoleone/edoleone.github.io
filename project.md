<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

## My Project

I applied machine learning techniques to investigate a dataset containing health data from 299 patients. In particular, I used Logistic Regression to build a predictive model for the risk of fatal heart attack. I then built a simple user interface where the user (i.e. a medical professional) can input patient-specific health data and determine whether said patient is at risk of a fatal heart attack. 

***

## Abstract 

Heart attacks are widely known to be the most common cause of death across most demographics. Hence, being able to predict the risk of a heart attack is extremely important in the medical field and models that successfully accomplish this can potentially save many lives. The following analysis of Chicco and Jurman's "Heart Failure Prediction" dataset [1] shows that heart failure can be predicted using a patient's age, ejection fraction, blood pressure (low/high), and serum creatinine value only. Including all features results in an arguable worse predictive model, since accuracy drops from 0.75 to 0.74 and precision drops from 0.73 to 0.63.

## Data

The dataset includes health data from 299 patients who all already had left ventricular disfunction and had a history of non-fatal heart failure [2]. There are input 12 features, and one output feature being whether the patient died of heart failure in the period between that appointment and the follow-up appointment. Here is a description of the input features:
* age: patient's age
* anaemia: decrease of red blood cells or hemoglobin (boolean)
* creatinine phosphokinase : level of the CPK enzyme in the blood (mcg/L)
* diabetes: if the patient has diabetes (boolean)
* ejection fraction: percentage of blood leaving the heart at each contraction, with 50-70% generally being considered as healthy
* high blood pressure: if the patient has hypertension (boolean)
* platelets: platelets in the blood (kiloplatelets/mL)
* serum creatinine: level of serum creatinine in the blood (mg/dL)
* serum sodium: level of serum sodium in the blood (mEq/L)
* sex: patient's sex (binary)
* smoking: whether patient smokes or not (boolean)
* time: follow up period in between appointments

A patient was considered to have anaemia if the haematocrit levels were lower than 36%. No clear description was given for how they differentiated between low and high blood pressure [2].

Table 1 below shows a table summary of the dataset:

|index|age|anaemia|creatinine\_phosphokinase|diabetes|ejection\_fraction|high\_blood\_pressure|platelets|serum\_creatinine|serum\_sodium|sex|smoking|time|DEATH\_EVENT|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|count|299\.0|299\.0|299\.0|299\.0|299\.0|299\.0|299\.0|299\.0|299\.0|299\.0|299\.0|299\.0|299\.0|
|mean|60\.83|0\.43|581\.84|0\.42|38\.08|0\.35|263358\.03|1\.39|136\.63|0\.65|0\.32|130\.26|0\.32|
|std|11\.89|0\.5|970\.29|0\.49|11\.83|0\.48|97804\.24|1\.03|4\.41|0\.48|0\.47|77\.61|0\.47|
|min|40\.0|0\.0|23\.0|0\.0|14\.0|0\.0|25100\.0|0\.5|113\.0|0\.0|0\.0|4\.0|0\.0|
|25%|51\.0|0\.0|116\.5|0\.0|30\.0|0\.0|212500\.0|0\.9|134\.0|0\.0|0\.0|73\.0|0\.0|
|50%|60\.0|0\.0|250\.0|0\.0|38\.0|0\.0|262000\.0|1\.1|137\.0|1\.0|0\.0|115\.0|0\.0|
|75%|70\.0|1\.0|582\.0|1\.0|45\.0|1\.0|303500\.0|1\.4|140\.0|1\.0|1\.0|203\.0|1\.0|
|max|95\.0|1\.0|7861\.0|1\.0|80\.0|1\.0|850000\.0|9\.4|148\.0|1\.0|1\.0|285\.0|1\.0|

Figure 1 below, instead, shows a visual depiction of each feature's distribution:

![](assets/IMG/img1.png){: width="500" }
*Figure 1: Histogram distribution of all features in the dataset.*

It must be mentioned that DEATH_EVENT, the target output variable, is slightly class imbalanced since there are twice as many patients who did not experience heart failure than patients who did.

## Preprocessing

Firstly, I converted all the non-integer data to integer. 

I then performed feature ranking for Logistic Regression. The results of the feature ranking showed that the 'time' feature dominated over all others by far. This is because a longer follow-up period, for a patient with pregressed heart issues, means a longer period of time during which new issues can arise. I then decided to remove the 'time' feature from the analysis. Since the aim of the project is to predict the chance of a fatal heart failure at the first appointment, it would be counterintuitive to include the length of the follow up period in the analysis.

## Modelling

For the machine learning model, I chose to use Logistic Regression since the target output variable is a boolean (classification problem). I considered using Random Forest Regression, but given the small size of the dataset, this resulted in very low R^2 scores. To contrast this, I tried increasing the maximum depth of the Random Forest, but this ultimately just resulted in the model overfitting the training data. I know that the model was overfitting the data because the difference between training and test R^2 scores was significant (>0.1).

I set the number of iteration at 20000 in order for all features to converge. I implemented k-fold cross validation in the model in order to make it more robust and reduce overfitting. I chose a 5 fold cross validation method, specifically. I thought about using a 10 fold model, given the small size of my dataset, but this resulted in the folds being extremely small, hence the 5 fold model choice.

After removing the 'time' feature, I performed feature ranking again. The method chosen for feature ranking was Recursive Feature Elimination (RFE), whereby one feature at the time is removed from the model with replacement and the features yielding the greatest negative impact when removed are deemed to be the most important ones. 

The standout features in terms of importance were ejection fraction and serum creatinine. Once the important features were identified, I dropped all other features from the dataset and I compared the confusion matrices and ROC curves of the model with all features and with the interest features only.

Finally, I built a simple user interface to make predictions using the model with the important features only.

## Results

Below is the list of features ranked from least important to most important based on the R^2 test score they produced when using the RFE method:
* Column: diabetes, Test R^2 score: 0.756
* Column: serum_sodium, Test R^2 score: 0.756
* Column: smoking, Test R^2 score: 0.753
* Column: anaemia, Test R^2 score: 0.749
* Column: high_blood_pressure, Test R^2 score: 0.746
* Column: creatinine_phosphokinase, Test R^2 score: 0.742
* Column: age, Test R^2 score: 0.739
* Column: platelets, Test R^2 score: 0.736
* Column: sex, Test R^2 score: 0.736
* Column: serum_creatinine, Test R^2 score: 0.709
* Column: ejection_fraction, Test R^2 score: 0.699

Figure 2 below shows a visual depiction of each feature's importance. In the histogram, shorter columns represent more important features, since the R^2 score was worse when these features were removed:

![](assets/IMG/img10.png){: width="500" }
*Figure 2: Feature importances.*

I then compared the performance of the model with all features included to the model with only ejection fraction and serum creatinine included. Below are the confusion matrices for both:

![](assets/IMG/img17.png){: width="500" }
*Figure 3: Confusion matrix for model with all features included.*

![](assets/IMG/img11.png){: width="500" }
*Figure 4: Confusion matrix for model with only ejection fraction and serum creatinine.*

The training and test R^2 scores for both models were:
* All features:
  * Training: 0.77
  * Test: 0.74
* Important features only:
  * Training: 0.75
  * Test: 0.75

Accuracy dropped from 0.75 to 0.74 going from the 2-feature model to the full model, and precision also dropped from 0.73 to 0.63.

I also plotted the ROC curve for both models, as shown below:

![](assets/IMG/img16.png){: width="500" }
*Figure 5: ROC curve for model with all features included.*

![](assets/IMG/img12.png){: width="500" }
*Figure 6: ROC curve for model with all features included.*

Finally, I used the following code snippet to create an interactive user interface for a medical professional to input patient data and obtain the DEATH_EVENT value. This was made using the model with important features only:
```python
serum_creatinine = float(input("Enter serum creatinine value: "))
ejection_fraction = float(input("Enter ejection fraction: "))

user_input = pd.DataFrame([[serum_creatinine, ejection_fraction]],
                           columns=['serum_creatinine', 'ejection_fraction'])  

# Make prediction
prediction = regr.predict(user_input)
predicted_death_event = np.where(prediction >= 0.5, 1, 0)[0]

print(f"Predicted Death Event: {predicted_death_event}")
```

Figure 7 and Figure 8 show examples of the model predicting a value of 1 for a patient with really poor values, and a value of 0 for a patient with relatively healthy values:

![](assets/IMG/img15.png){: width="500" }
*Figure 7: Predictive model making a prediction of 1.*

![](assets/IMG/img14.png){: width="500" }
*Figure 8: Predictive model making a prediction of 0.*

## Discussion

Seeing that the test R^2 score was higher for the 2-feature model than for the full model (0.75 vs 0.74), it can be inferred that the 2-feature model's predictions match the data better than the full model's predictions. 

Nonetheless, the Area Under the Curve on the ROC curve was larger for the full model (0.84 vs 0.78). This might suggest that the full model has better classification ability. However, as seen in Figure 5 and Figure 6, the ROC curves were very jagged for this particular dataset, indicating their poor reliability. This is most likely to be attributed to the small size of the dataset.

Given the nature and applications of this model, the most important factor when judging the model's performance is keeping the number of False Negative predictions low. As such, it might be a good idea to lower the threshold for a DEATH_EVENT prediction of 1 from 0.5 to a lower value. 

## Conclusion

In conclusion, the 2-feature model performed better than the full model in terms of accuracy, precision, and R^2 test score. Having an efficient 2-feature model would also be more practical for real-world applications as it would allow for quicker diagnoses and immediate preventive treatment. Given a larger dataset of the same type, it would be interesting to try to predict the proper length of follow-up period given the patient's health data, perhaps adding in a risk factor.


## References

[1] “Heart Failure Prediction.” Kaggle, 20 June 2020, www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data/data?select=heart_failure_clinical_records_dataset.csv.

[2] Chicco, D., Jurman, G. Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. BMC Med Inform Decis Mak 20, 16 (2020). https://doi.org/10.1186/s12911-020-1023-5

[back](./)


