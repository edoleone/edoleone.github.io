<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

## My Project

I applied machine learning techniques to investigate a dataset containing health data from 299 patients. In particular, I used Logistic Regression to build a predictive model for the risk of fatal heart attack. I then built a simple user interfce where the user (i.e. a medical professional) can input patient-specific health data and determine whether said patient is at risk of a fatal heart attack. 

***

## Abstract 

Heart attacks are widely known to be the most common cause of death across most demographics. Hence, being able to predict the risk of a heart attack is extremely important in the medical field and models that successfully accomplish this can potentially save many lives. The following analysis of Chicco and Jurman's "Heart Failure Prediction" dataset [1] shows that heart failure can be predicted using a patient's age, ejection fraction, blood pressure (low/high), and serum creatinine value only. Including all features results in an arguable worse predictive model, since accuracy drops from 0.78 to 0.75 and recall drops from 0.50 to 0.44.

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

Table 1 below shows a table summary of the dataset:
|index|age|anaemia|creatinine\_phosphokinase|diabetes|ejection\_fraction|high\_blood\_pressure|platelets|serum\_creatinine|serum\_sodium|sex|smoking|time|DEATH\_EVENT|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|count|299\.0|299\.0|299\.0|299\.0|299\.0|299\.0|299\.0|299\.0|299\.0|299\.0|299\.0|299\.0|299\.0|
|mean|60\.82943143812709|0\.431438127090301|581\.8394648829432|0\.4180602006688963|38\.08361204013378|0\.3511705685618729|263358\.02675585286|1\.3938795986622072|136\.62541806020067|0\.6488294314381271|0\.3210702341137124|130\.2608695652174|0\.3210702341137124|
|std|11\.894996551670072|0\.49610726813307915|970\.2878807124363|0\.49406706510360887|11\.834840741039173|0\.4781363790627452|97804\.23686859861|1\.034510064089853|4\.412477283909233|0\.47813637906274487|0\.4676704280567721|77\.61420795029342|0\.4676704280567721|
|min|40\.0|0\.0|23\.0|0\.0|14\.0|0\.0|25100\.0|0\.5|113\.0|0\.0|0\.0|4\.0|0\.0|
|25%|51\.0|0\.0|116\.5|0\.0|30\.0|0\.0|212500\.0|0\.9|134\.0|0\.0|0\.0|73\.0|0\.0|
|50%|60\.0|0\.0|250\.0|0\.0|38\.0|0\.0|262000\.0|1\.1|137\.0|1\.0|0\.0|115\.0|0\.0|
|75%|70\.0|1\.0|582\.0|1\.0|45\.0|1\.0|303500\.0|1\.4|140\.0|1\.0|1\.0|203\.0|1\.0|
|max|95\.0|1\.0|7861\.0|1\.0|80\.0|1\.0|850000\.0|9\.4|148\.0|1\.0|1\.0|285\.0|1\.0|

Figure 1 below, instead, shows a visual depiction of each feature's distribution:
![](assets/IMG/datapenguin.png){: width="500" }
*Figure 1: Here is a caption for my diagram. This one shows a pengiun [1].*

## Preprocessing

## Modelling

Here are some more details about the machine learning approach, and why this was deemed appropriate for the dataset. 

<p>
When \(a \ne 0\), there are two solutions to \(ax^2 + bx + c = 0\) and they are
  \[x = {-b \pm \sqrt{b^2-4ac} \over 2a}.\]
</p>

The model might involve optimizing some quantity. You can include snippets of code if it is helpful to explain things.

```python
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_classification
X, y = make_classification(n_features=4, random_state=0)
clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
clf.fit(X, y)
clf.predict([[0, 0, 0, 0]])
```

This is how the method was developed.

## Results

Figure X shows... [description of Figure X].

## Discussion

From Figure X, one can see that... [interpretation of Figure X].

## Conclusion

Here is a brief summary. From this work, the following conclusions can be made:
* first conclusion
* second conclusion

Here is how this work could be developed further in a future project.

## References
[1] “Heart Failure Prediction.” Kaggle, 20 June 2020, www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data/data?select=heart_failure_clinical_records_dataset.csv.
[2] Chicco, D., Jurman, G. Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. BMC Med Inform Decis Mak 20, 16 (2020). https://doi.org/10.1186/s12911-020-1023-5
[3]

[back](./)

## Notes on stuff to add
* discussion on why time was removed
* discussion on r^2 to check for overfitting
* effect of changing threshold on increasing recall
* how feature ranking was obtained for LR
* why choosing number of iterations
* why the ROC is jagged
* why 5 folds for kfold cross validation
* discussion of area under the curve for ROC
* extension on calculating proper follow up period with risk factor if given a larger dataset

