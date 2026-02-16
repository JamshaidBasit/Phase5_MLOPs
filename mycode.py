import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data=pd.read_csv("house_data.csv")
print(data.head())

x=data[["Age","Bedrooms","Area"]]
y=data["Price"]

X_train,x_test, Y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(X_train, Y_train)

prediction=model.predict(x_test)
print(prediction)