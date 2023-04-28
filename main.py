import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,ElasticNet
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sn

data = pd.read_csv("HousingData.csv")
data.head() #查看前五行
data.tail()
data.sample()
data.info() #查看数据的类型，完整性
data.describe() #查看数据的统计特征（均值、方差等）
data.dropna(inplace=True) #删除有缺失的样本

for id in data.columns[:-1]:
    sn.pairplot(data[[id,data.columns[-1]]])
y = data['MEDV'] # 标签-房价
X = data.drop(['MEDV'], axis=1) #去掉标签（房价）的数据子集

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
scaler=preprocessing.StandardScaler().fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

lr = LinearRegression() #实例化一个线性回归对象
lr.fit(X_train, y_train) #采用fit方法，拟合回归系数和截距
print(lr.intercept_)   #输出截距
print(lr.coef_)   #输出系数   可分析特征的重要性以及与目标的关系
y_pred = lr.predict(X_test)#模型预测
print("R2=",r2_score(y_test, y_pred))#模型评价, 决定系数
#print("mse=",mean_squared_error(y_test, y_pred))#均方误差
#print(lr.intercept_)  #输出截距
#print(lr.coef_)  #系数

plt.plot(y_test.values,c="r",label="y_test")
plt.plot(y_pred,c="b",label="y_pred")
plt.legend()