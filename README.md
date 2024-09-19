# EX.NO:4-DS
## AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

## ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

## FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

## FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

## CODING AND OUTPUT:
```
Developed by :RASIKA M
Reference number :212222230117
```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data
```
![image](https://github.com/22008837/EXNO-4-DS/assets/120194155/af9d6cb5-d78d-4e6d-a0a5-c4fefa8d873e)
```
data.isnull().sum()
```
![image](https://github.com/22008837/EXNO-4-DS/assets/120194155/42e0ee9d-9060-4a17-b449-497ecbfc15f6)
```
missing=data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/22008837/EXNO-4-DS/assets/120194155/6b99e1bc-b553-454a-bb19-6b10e040640e)
```
data2=data.dropna(axis=0)
data2
```
![image](https://github.com/22008837/EXNO-4-DS/assets/120194155/0810aafc-19fc-4357-aacb-e4cd067acaa1)
```
sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/22008837/EXNO-4-DS/assets/120194155/28ed40fe-f8ac-417f-a1c8-516f1b91b313)
```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/22008837/EXNO-4-DS/assets/120194155/87acafa4-a7c2-4ad8-b6d1-d3d37c68aef1)
```
data2
```
![image](https://github.com/22008837/EXNO-4-DS/assets/120194155/c1d8a9b7-3cd8-41d1-9b95-f605cb7f2f6b)
```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![image](https://github.com/22008837/EXNO-4-DS/assets/120194155/50d44c2c-74e4-43a4-8254-8d5edbf90c2e)
```
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/22008837/EXNO-4-DS/assets/120194155/a608bb9f-73cc-4e9b-b510-9fa283e29938)
```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/22008837/EXNO-4-DS/assets/120194155/de00a9ed-032f-4021-a8da-4e5b40383b16)
```
y=new_data['SalStat'].values
print(y)
```
![image](https://github.com/22008837/EXNO-4-DS/assets/120194155/ba349b73-0abf-4841-8a0f-fe83c51a290f)
```
x=new_data[features].values
print(x)
```
![image](https://github.com/22008837/EXNO-4-DS/assets/120194155/d66194e7-022c-4f66-aa80-3c58f7cb690c)
```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```
![image](https://github.com/22008837/EXNO-4-DS/assets/120194155/bdedb851-dd59-440e-9a52-ca0c1aeb696b)
```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![image](https://github.com/22008837/EXNO-4-DS/assets/120194155/9548fe34-d8e5-465b-8851-a2d565f6f4ff)
```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![image](https://github.com/22008837/EXNO-4-DS/assets/120194155/b870f557-5b08-46ae-b18e-4ddcaea38f65)
```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
![image](https://github.com/22008837/EXNO-4-DS/assets/120194155/7b7ee3c1-48c9-4cf9-8caa-7b68d4664bde)
```
data.shape
```
![image](https://github.com/22008837/EXNO-4-DS/assets/120194155/4928fb04-5583-4dd0-b371-286fa0faf4d5)
```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/22008837/EXNO-4-DS/assets/120194155/66f5e095-dbd9-49a8-a02f-019c01e79381)
```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/22008837/EXNO-4-DS/assets/120194155/79bc78e8-681d-432f-a4da-e931f0c72f9d)
```
tips.time.unique()
```
![image](https://github.com/22008837/EXNO-4-DS/assets/120194155/a15cea1d-8940-48c2-988b-39211ba5d9f9)
```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![image](https://github.com/22008837/EXNO-4-DS/assets/120194155/e47a3319-e86f-4ba5-bd2d-724c491c04f3)
```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
![image](https://github.com/22008837/EXNO-4-DS/assets/120194155/ba34761d-08fc-4d1c-83f4-e8fb96d25cfa)

## RESULT:
Thus, Feature selection and Feature scaling has been used on thegiven dataset.
