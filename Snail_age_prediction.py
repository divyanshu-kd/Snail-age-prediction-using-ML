import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from numpy.core.fromnumeric import size
import matplotlib.pyplot as plt
from pandas.io.formats import style
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
LR = LinearRegression()

# Q1
print("Simple Linear Regression")
df = pd.read_csv("abalone.csv")#reading csv file
cor = df.corr(method="pearson").iloc[:7, 7:]#finding correlation of data
print("Corr of target attribute with other attribute as follows :")
print(cor)

x = df["Shell weight"]#taking attribute shell weight as it has max correlation
y = df["Rings"]
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=.3,random_state=42)#spiliting data 


LR.fit(x_train.values.reshape(-1, 1), y_train.values)#fitting data in linear regression model
pre_train = LR.predict(x_train.values.reshape(-1, 1))

plt.scatter(x_train, y_train, label="Train data",color='red', marker='3', alpha=1)#plotting scatter plot between xtrain and ytarin
plt.plot(x_train, pre_train, label="Best-fit line", color="blue", linewidth=3)#plotting best fit line in the graph
plt.xlabel("X-train", size=16)
plt.ylabel("Y-train", size=16)
plt.legend()
plt.show()



#calculating accuracy score
y_pred = LR.predict(x_train.values.reshape(-1, 1))
err = (np.sqrt(mean_squared_error(y_train, y_pred)))
tot = y_train.mean()
acc = round((1-(err/tot))*100,3)
print("RMSE of prediction for training data is :", round(err,3))
print("Accuracy of prediction for training data is :", acc, "%")

y_pred = LR.predict(x_test.values.reshape(-1, 1))
err = (np.sqrt(mean_squared_error(y_test, y_pred)))
tot = y_test.mean()
acc = round((1-(err/tot))*100,3)
print("RMSE of prediction for testing data is :", round(err,3))
print("Accuracy of prediction for testing data is :", acc, "%")

LR.fit(x_train.values.reshape(-1, 1), y_train.values)
pre_train_rings = LR.predict(x_train.values.reshape(-1, 1))

plt.scatter(y_train, pre_train_rings, marker='o', color='g')
plt.title("Actual Rings vs predicted", size=22)
plt.xlabel("Actual NO. of rings", size=22)
plt.ylabel("Predicted NO. of rings", size=22)
plt.show()


# Q2

print()
print("Multivariate linear regression")
x = df.loc[::, :"Shell weight"]
y = df["Rings"]
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=.3,random_state=42)
LR.fit(x_train.values, y_train.values)
y_pred = LR.predict(x_train.values)

#repeating steps as in Q1
y_pred = LR.predict(x_train.values)
err = (np.sqrt(mean_squared_error(y_train, y_pred)))
tot = y_train.mean()
acc = round((1-(err/tot))*100,3)

print("RMSE of prediction for training data is :", round(err,3))
print("Accuracy of prediction for training data is :", acc, "%")

y_pred = LR.predict(x_test.values)
err = (np.sqrt(mean_squared_error(y_test, y_pred)))
tot = y_test.mean()
acc = round((1-(err/tot))*100,3)

print("RMSE of prediction for testing data is :", round(err,3))
print("Accuracy of prediction for testing data is :", acc, "%")

LR.fit(x_train.values, y_train.values)
pre_train_rings = LR.predict(x_train.values)

plt.scatter(y_train, pre_train_rings, color='blue')
plt.title("Actual Rings vs predicted", size=22)
plt.xlabel("Actual NO. of rings", size=16)
plt.ylabel("Predicted NO. of rings", size=16)
plt.grid(b=True, color='black', linestyle='-.', linewidth=0.5, alpha=0.7)
plt.show()

# Q3
print()
print("Simple Polynomial regression")
print()


x = df["Shell weight"].values#attribute with highest pearson coefficint
y = df["Rings"].values
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=.3,random_state=42)
err1 = []

for i in range(2, 6):

    poly_features = PolynomialFeatures(i)#
    x_poly = poly_features.fit_transform(x_train.reshape(-1, 1))
    LR.fit(x_poly, y_train.reshape(-1, 1))
    y_pred = LR.predict(x_poly)
   
    err1.append(np.sqrt(mean_squared_error(y_train.reshape(-1, 1), y_pred)))
    
    err = np.sqrt(mean_squared_error(y_train.reshape(-1, 1), y_pred))
    tot = y_train.mean()
    acc = round((1-(err/tot))*100,3)
    print("RMSE for train data for p = {}".format(i), "is :", round(err,3))
    print("Accuracy percent for train data for p = {}".format(i), "is :", acc, "%")

plt.bar([2, 3, 4, 5], err1, color=["green", "yellow", "blue", "red"])
plt.title("RMSE for diff values of P (Training data)", size=22)
plt.show()

plt.scatter(x_train, y_train)
plt.xlabel("X-train", size=16)
plt.ylabel("Y-train", size=16)
plt.title("Scatterplot along with best fit curve", size=22)

plt.scatter(x_train, y_pred, color="red", linewidth=3)
plt.grid(b=True, color='black', linestyle='-.', linewidth=0.5, alpha=0.9)
plt.show()

err2 = []
print()
for i in range(2, 6):

    poly_features = PolynomialFeatures(i)
    x_poly = poly_features.fit_transform(x_test.reshape(-1, 1))
    LR.fit(x_poly, y_test.reshape(-1, 1))
    y_pred = LR.predict(x_poly)

   
    err2.append(np.sqrt(mean_squared_error(y_test.reshape(-1, 1), y_pred)))
    err = np.sqrt(mean_squared_error(y_test.reshape(-1, 1), y_pred))
    tot = y_test.mean()
    acc = round((1-(err/tot))*100,3)
    print("RMSE for test data for p = {}".format(i), "is :", round(err,3))
    print("Accuracy percent for test data for p = {}".format(i), "is :", acc, "%")

plt.bar([2, 3, 4, 5], err2, color=["y", "b", "g", "r"])
plt.title("RMSE for diff values of P (Testing data)", size=22)
plt.show()

plt.scatter(y_test, y_pred)
plt.title("Actual Rings vs predicted", size=22)
plt.xlabel("Actual NO. of rings", size=16)
plt.ylabel("Predicted NO. of rings", size=16)

plt.grid(b=True, color='black', linestyle='-.', linewidth=0.5, alpha=0.9)
plt.show()

# Q4
print()
print("Multivariate Polynomial regression")
print()
err3 = []
x = df.loc[::, :"Shell weight"]
y = df["Rings"]
x_train, x_test, y_train, y_test = train_test_split(x, y)

for i in range(2, 6):

    poly_features = PolynomialFeatures(i)
    x_poly = poly_features.fit_transform(x_train)
    LR.fit(x_poly, y_train.values.reshape(-1, 1))
    y_pred = LR.predict(x_poly)
    err3.append(np.sqrt(mean_squared_error(y_train, y_pred)))
    
    err = np.sqrt(mean_squared_error(y_train, y_pred))
    tot = y_train.mean()
    acc = round((1-(err/tot))*100,3)
    print("RMSE for train data for p = {}".format(i), "is :", round(err,3))
    print("Accuracy percent for train data for p = {}".format(i), "is :", acc, "%")



plt.bar([2, 3, 4, 5], err3, color=["red", "green", "black", "yellow"])
plt.title("RMSE for diff values of P (Training data)", size=22)
plt.show()

err4 = []
print()
for i in range(2, 6):

    poly_features = PolynomialFeatures(i)
    x_poly = poly_features.fit_transform(x_test)
    LR.fit(x_poly, y_test.values.reshape(-1, 1))
    y_pred = LR.predict(x_poly)

    err4.append(np.sqrt(mean_squared_error(y_test, y_pred)))
    
    err = np.sqrt(mean_squared_error(y_test, y_pred))
    tot = y_test.mean()
    acc = round((1-(err/tot))*100,3)
    print("RMSE for test data for p = {}".format(i), "is :", round(err,3))
    print("Accuracy percent for test data for p = {}".format(i), "is :", acc, "%")



plt.bar([2, 3, 4, 5], err4, color=["yellow", "blue", "green", "red"])
plt.title("RMSE for diff values of P (Testing data)", size=22)
plt.show()

plt.scatter(y_test, y_pred, color="y")
plt.title("Actual Rings vs predicted", size=22)
plt.xlabel("Actual NO. of rings", size=16)
plt.ylabel("Predicted NO. of rings", size=16)
plt.grid(b=True, color='black', linestyle='-.', linewidth=0.5, alpha=0.9)
plt.show()
