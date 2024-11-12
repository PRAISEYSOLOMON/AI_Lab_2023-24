# Ex.No: 10 Learning – Use Supervised Learning  
### DATE:                                                                            
### REGISTER NUMBER : 212222040117
### AIM: 
To write a program to train the model to predict the breast cancer.
###  Algorithm:

1. Load the breast cancer dataset.
2. Split the data into training and testing.
3. Train the model bt logistic regression method.
4. Use the trained model to predict cancer diagnosis on the test set.
5. Calculate accuracy by comparing the predictions to the true values.

### Program:
```
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
%matplotlib inline
df = pd.read_csv('data.csv')
df.head()
df.shape
df.info()
df.columns
df.isnull().sum()
df.drop("Unnamed: 32", axis=1, inplace=True)
df.drop('id',axis=1, inplace=True)
df.describe()
df['diagnosis'].value_counts()
df['diagnosis']=df['diagnosis'].map({'B':0,'M':1})
df['diagnosis'].value_counts()
from sklearn.model_selection import train_test_split

# splitting data
X_train, X_test, y_train, y_test = train_test_split(
                df.drop('diagnosis', axis=1),
                df['diagnosis'],
                test_size=0.2,
                random_state=42)

print("Shape of training set:", X_train.shape)
print("Shape of test set:", X_test.shape)
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
predictions1 = logreg.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report

print("Confusion Matrix: \n", confusion_matrix(y_test, predictions1))
print('\n')
print(classification_report(y_test, predictions1))
from sklearn.metrics import accuracy_score

logreg_acc = accuracy_score(y_test, predictions1)
print("Accuracy of the Logistic Regression Model is: ", logreg_acc)
```

### Output:


### Result:
Thus the system was trained successfully and the prediction was carried out.
