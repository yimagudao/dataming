@@ -0,0 +1,58 @@
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 09:34:54 2018

@author: moulf
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression,LogisticRegression,Ridge,SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt

#pandas加载数据
data = pd.read_csv(r'data\Folds5x2_pp.csv')
data = data.dropna()
data = data.drop_duplicates()
X = data.iloc[:,0:4]
y = data.iloc[:,4]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,random_state= 250)

#标准化
stdsc = StandardScaler()
X_train_conti_std = stdsc.fit_transform(X_train[['AT', 'V', 'AP', 'RH']])
X_test_conti_std = stdsc.fit_transform(X_test[['AT', 'V', 'AP', 'RH']])
# 将ndarray转为dataframe
X_train = pd.DataFrame(data=X_train_conti_std, columns=['AT', 'V', 'AP', 'RH'], index=X_train.index)
X_test = pd.DataFrame(data=X_test_conti_std, columns=['AT', 'V', 'AP', 'RH'], index=X_test.index)

#线性回归
reg = LinearRegression()
#随机梯度下降
#reg = SGDRegressor()
#岭回归
#reg = Ridge()

reg.fit(X_train,y_train)
print('intercept:',reg.intercept_)
print('coef:',reg.coef_)

y_pred = reg.predict(X_test)
#confusion_matrix = confusion_matrix(y_test.astype('int'), y_pred.astype('int'))

plt.scatter(y_test,y_pred,color='green')
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],'k--',color='blue',linewidth=3)
plt.xlabel('True')
plt.ylabel('Prediect')
plt.show()

print(reg.score(X_test,y_test))
print("MSE:",mean_squared_error(y_test,y_pred))

predicted = cross_val_predict(reg, X, y, cv=10)
print("10折交叉验证后 MSE:",mean_squared_error(y,predicted))