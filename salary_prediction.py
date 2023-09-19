import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# regression model for salary prediction based on years of experience
# algo used : Linear regression
# Yi=B0+B1Xi is linear regression

data=pd.read_csv("/kaggle/input/salary-dataset-simple-linear-regression/Salary_dataset.csv")
#print(data)

x=data["YearsExperience"].to_numpy().reshape(-1,1)
y=data["Salary"].to_numpy().reshape(-1,1)

#training data
x_train=x[:-5]
y_train=y[:-5]

#testing data
x_test=x[-5:]
y_test=y[-5:]

LRM=LinearRegression()
LRM.fit(x_train,y_train);
predicted=LRM.predict(x_test)

print(f"predicted vals : {predicted}")
print(f"actual vals : {y_test} ")


print(f"training accuracy : {np.floor(LRM.score(x_train,y_train)*100)}%")
mse=mean_squared_error(y_test,predicted)
mae=mean_absolute_error(y_test,predicted)
#lower the values of mse and mae better performance

r2=r2_score(y_test,predicted) # higher the value better the performance
print(f"mean sqrd error : {np.floor(mse)}")
print(f"mean abs error : {np.floor(mae)}")
print(f"R sqrd : {r2}")

#plotting
plt.scatter(x_test,y_test,c="yellow")
plt.plot(x_test,predicted,c="green")
plt.xlabel("years of exp.")
plt.ylabel("salary")
#plt.figure(25
plt.legend(["actual vals","predicted vals"])
plt.show()
