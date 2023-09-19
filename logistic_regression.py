import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

#credit card fraud detection model 
#algo used : Logistic Regression
#Log Reg. is used for classification 


data=pd.read_csv("/kaggle/input/creditcardcsv/creditcard.csv",index_col=0)
X,Y=data.drop(["Class"],axis=1),data["Class"]

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3)
LgRM=LogisticRegression(max_iter=150000)
LgRM.fit(x_train,y_train)

predicted=LgRM.predict(x_test)
print(predicted)


#data.info()
