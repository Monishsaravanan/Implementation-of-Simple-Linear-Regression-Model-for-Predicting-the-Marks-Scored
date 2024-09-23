# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step1: Start the program.

Step2: Import the standard Libraries. 

Step3: Set variables for assigning dataset values.

Step4: Import linear regression from sklearn.

Step5: Assign the points for representing in the graph.

Step6: Predict the regression for marks by using the representation of the graph.

Step7: Compare the graphs and hence we obtained the linear regression for the given datas.

Step8: Stop the program.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: MONISH S
RegisterNumber: 212223040115
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("C:/Users/admin/Downloads/student_scores.csv")
df.head()

df.tail()

X=df.iloc[:,:-1].values
X

Y=df.iloc[:,1].values
Y

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

Y_pred

Y_test

plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_train,Y_train,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="yellow")
plt.title("Hours vs Scores(Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```
<br><br>
## Output:

![image](https://github.com/user-attachments/assets/0fc6ed3c-a3cc-48e2-9402-e394d0bef64e)

![image](https://github.com/user-attachments/assets/9da1756c-cf2f-4aed-8aa0-fcb10b176554)

![image](https://github.com/user-attachments/assets/da2e1286-5c6c-4aa5-95f4-392f0c615729)

![image](https://github.com/user-attachments/assets/ce6aaa9e-7fd9-473b-83a9-a617ac6fee5b)

![image](https://github.com/user-attachments/assets/0a74d7b0-be88-4274-94bf-936d9879adf2)

![image](https://github.com/user-attachments/assets/88a247fb-1711-4c1c-9a85-e3f21b1531c3)

![image](https://github.com/user-attachments/assets/3d0eea02-b7dd-48e6-a85a-070c7b754ed0)

![image](https://github.com/user-attachments/assets/b080e329-73f9-4594-bdf1-045b02f67201)

![image](https://github.com/user-attachments/assets/69bf284c-782e-44ee-b65c-5e75b1a13467)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
