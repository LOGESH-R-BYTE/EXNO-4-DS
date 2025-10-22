# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
import pandas as pd

import numpy as np

df=pd.read_csv("bmi.csv")

df


<img width="594" height="657" alt="image" src="https://github.com/user-attachments/assets/9b3754bb-c040-4f6f-90c4-4d5b3e01009f" />

df.head()

<img width="556" height="329" alt="image" src="https://github.com/user-attachments/assets/de464da8-5b99-4c81-961b-b1534be75e59" />


df.dropna()

<img width="644" height="642" alt="image" src="https://github.com/user-attachments/assets/3d1d62aa-a918-4210-a78b-b5b26daf236b" />


max_vals=np.max(np.abs(df[['Height','Weight']]))

max_vals

<img width="265" height="78" alt="image" src="https://github.com/user-attachments/assets/510d1fa4-d8c7-4789-9a2e-079f293931ee" />


from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()

df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])

df.head(10)

<img width="569" height="574" alt="image" src="https://github.com/user-attachments/assets/3a8094ee-e588-4546-888c-3c6a0ae18a6e" />

df1=pd.read_csv("bmi.csv")

df2=pd.read_csv("bmi.csv")

df3=pd.read_csv("bmi.csv")

df4=pd.read_csv("bmi.csv")

df5=pd.read_csv("bmi.csv")

df1

<img width="547" height="641" alt="image" src="https://github.com/user-attachments/assets/6de40f8e-4aaa-4ac1-9c45-06ef65c17ad0" />


from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])

df.head(10)

<img width="593" height="540" alt="image" src="https://github.com/user-attachments/assets/ebfee71a-a324-4440-8d02-49407a1a8c43" />

from sklearn.preprocessing import Normalizer

scaler=Normalizer()

df2[['Height','Weight']]=scaler.fit_transform(df2[['Height','Weight']])

df2

<img width="594" height="636" alt="image" src="https://github.com/user-attachments/assets/cc4d7676-ccc4-459f-bbc6-3961bc877684" />


from sklearn.preprocessing import MaxAbsScaler

max1=MaxAbsScaler()

df3[['Height','Weight']]=max1.fit_transform(df3[['Height','Weight']])

df3


<img width="554" height="655" alt="image" src="https://github.com/user-attachments/assets/dd80a3b8-9526-4fe4-baf8-2bc1b8517bbf" />


from sklearn.preprocessing import RobustScaler

roub=RobustScaler()

df4[['Height','Weight']]=roub.fit_transform(df4[['Height','Weight']])

df4

<img width="561" height="643" alt="image" src="https://github.com/user-attachments/assets/f17339f1-e8d2-4b89-a63d-d19ad74da835" />


from sklearn.feature_selection import SelectKBest,f_regression,mutual_info_classif

from sklearn.feature_selection import chi2

data=pd.read_csv("income(1) (1).csv")

data

<img width="817" height="237" alt="image" src="https://github.com/user-attachments/assets/f7d94d56-d909-44cb-b565-98bba9ba807b" />


data1=pd.read_csv('/content/titanic_dataset (1).csv')

data1

<img width="825" height="331" alt="image" src="https://github.com/user-attachments/assets/fd9bb0af-2474-4d68-84a0-e3ee73c35c57" />


data1=data1.dropna()

x=data1.drop(['Survived','Name','Ticket'],axis=1)

y=data1['Survived']

data1['Sex']=data1['Sex'].astype('category')

data1['Cabin']=data1['Cabin'].astype('category')

data1['Embarked']=data1['Embarked'].astype('category')



data1['Sex']=data1['Sex'].cat.codes

data1['Cabin']=data1['Cabin'].cat.codes

data1['Embarked']=data1['Embarked'].cat.codes

data1

<img width="824" height="317" alt="image" src="https://github.com/user-attachments/assets/53ad9678-d388-476e-a457-64f978596b2f" />


k=5

selector=SelectKBest(score_func=chi2,k=k)

x=pd.get_dummies(x)

x_new=selector.fit_transform(x,y)

x_encoded=pd.get_dummies(x)

selector=SelectKBest(score_func=chi2,k=5)

x_new=selector.fit_transform(x_encoded,y)


selected_feature_indices=selector.get_support(indices=True)

selected_features=x.columns[selected_feature_indices]

print("Selected Features:")

print(selected_features)

<img width="814" height="64" alt="image" src="https://github.com/user-attachments/assets/5a1af28d-968b-4527-b526-aa03dbc992ca" />


selector=SelectKBest(score_func=f_regression,k=5)

x_new=selector.fit_transform(x_encoded,y)

selected_feature_indices=selector.get_support(indices=True)

selected_features=x.columns[selected_feature_indices]

print("Selected Features:")

print(selected_features)

<img width="821" height="66" alt="image" src="https://github.com/user-attachments/assets/56fd7713-a0fc-461e-87fd-db301af7caae" />

selector=SelectKBest(score_func=mutual_info_classif,k=5)

x_new=selector.fit_transform(x,y)

selected_feature_indices=selector.get_support(indices=True)

selected_features=x.columns[selected_feature_indices]

print("Selected Features:")

print(selected_features)

<img width="821" height="87" alt="image" src="https://github.com/user-attachments/assets/f553781d-b368-4314-a53e-1ad683e7b5a2" />


from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier()

sfm=SelectFromModel(model,threshold='mean')

x=pd.get_dummies(x)

sfm.fit(x,y)

selected_features=x.columns[sfm.get_support()]

print("Selected Features:")

print(selected_features)

<img width="812" height="122" alt="image" src="https://github.com/user-attachments/assets/eab15c83-6543-43a8-82e7-1ee418ebb1f9" />

from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier(n_estimators=100,random_state=42)

model.fit(x,y)

feature_selection=model.feature_importances_

threshold=0.1

selected_features=x.columns[feature_selection>threshold]

print("Selected Features:")

print(selected_features)

<img width="817" height="83" alt="image" src="https://github.com/user-attachments/assets/ebb993b8-c832-4332-ab27-19ed22e05b8d" />


model=RandomForestClassifier(n_estimators=100,random_state=42)

model.fit(x,y)

feature_importance=model.feature_importances_

threshold=0.15

selected_features=x.columns[feature_importance>threshold]

print("Selected Features:")

print(selected_features)

<img width="447" height="77" alt="image" src="https://github.com/user-attachments/assets/501ba72f-7ae6-40a7-9213-fa4042ddff7b" />


# RESULT:

Thus, Feature selection and Feature scaling has been used on thegiven dataset.
       
