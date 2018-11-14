from sklearn.preprocessing import StandardScaler
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA

import pandas as  pd
import matplotlib.pyplot as plt
import numpy as np 


import warnings
warnings.filterwarnings('ignore') #抑制warning输出

stdsc = StandardScaler()
pca = PCA(n_components=30)
def prepare(path):

    data = pd.read_csv(path)
    if 'id' in data.columns: 
            data.pop('id')
    data.pop('居住状态') #缺少89.75%的数据，直接弹出，没什么用
    data.pop('出租方式') #同理弹出
    data.pop('装修情况')
    data.pop('房屋朝向')
    if('月租金' in data.columns):
       data.pop('月租金')
   # data = pd.get_dummies(data)
 #   print(data.shape)
    values={'区':12.0,
        '小区房屋出租数量':0.08203125,
        '位置':52,                         #位置采取的是最频繁策略
        '地铁线路':0,                      #地铁线路为nan的时候，地铁站点、距离都是nan所以认为是没有
        '地铁站点':0,                      #地铁附近，所以全部设置为0
        '距离':0}
    data=data.fillna(value=values)#分别对应处理缺失值
  #  print(data.columns)
   # data = pca.fit_transform(data)
    return  stdsc.fit_transform(data)

data = pd.DataFrame(prepare("train.csv"))
target = pd.read_csv("train.csv").iloc[:,18]
x_train,x_test,y_train,y_test = train_test_split(data.iloc[:,:data.shape[1]],
                                                target,#stdsc.inverse_transform(data.iloc[:,data.shape[1]-1]),
                                                test_size=0.2,
                                                random_state=0)

def try_different_method(model):
    model.fit(x_train,y_train)
    score = model.score(x_test, y_test)
    result = model.predict(x_test)
    print(model.__class__.__name__)
    print("meg: ",mean_squared_error(y_test,result))
    plt.figure()
    plt.plot(np.arange(len(result)), y_test,'go-',label='true value')
    plt.plot(np.arange(len(result)),result,'ro-',label='predict value')
    plt.title('score: %f'%score)
    plt.legend()
    plt.show()

#尝试使用树回归

# model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=20)

from sklearn import tree
model_DecisionTreeRegressor = tree.DecisionTreeRegressor()
####3.2线性回归####
from sklearn import linear_model
model_LinearRegression = linear_model.LinearRegression()
####3.3SVM回归####
from sklearn import svm
model_SVR = svm.SVR()
####3.4KNN回归####
from sklearn import neighbors
model_KNeighborsRegressor = neighbors.KNeighborsRegressor()
####3.5随机森林回归####
from sklearn import ensemble
model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=20)#这里使用20个决策树
####3.6Adaboost回归####
from sklearn import ensemble
model_AdaBoostRegressor = ensemble.AdaBoostRegressor(n_estimators=50)#这里使用50个决策树
####3.7GBRT回归####
from sklearn import ensemble
model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(n_estimators=100)#这里使用100个决策树
####3.8Bagging回归####
from sklearn.ensemble import BaggingRegressor
model_BaggingRegressor = BaggingRegressor()
####3.9ExtraTree极端随机树回归####
from sklearn.tree import ExtraTreeRegressor
model_ExtraTreeRegressor = ExtraTreeRegressor(splitter='best',max_depth=50,max_features=12)


#使用极端树
# x_train = data.iloc[:,:data.shape[1]-1]
# y_train = data.iloc[:,data.shape[1]-1]
# pca.fit(x_train)
# x_train = pca.transform(x_train)

# model_ExtraTreeRegressor.fit(x_train,y_train)
# data = pd.DataFrame(prepare("test.csv"))
# # data = pca.transform(data)

# y_p = model_ExtraTreeRegressor.predict(data)
# df = pd.DataFrame({'id':list(range(1,y_p.shape[0]+1)),'price':y_p})
# df.to_csv('tree.csv',index=0)
###########4.具体方法调用部分##########
# try_different_method(model_DecisionTreeRegressor)
# try_different_method(model_LinearRegression)
# try_different_method(model_RandomForestRegressor)
# try_different_method(model_AdaBoostRegressor)#不行 1.06
# try_different_method(model_GradientBoostingRegressor) # 1.026
# try_different_method(model_BaggingRegressor) #0.445
# try_different_method(model_ExtraTreeRegressor) #1.08  spliter best 0.405

# x_train = pca.fit_transform(x_train)
# x_test = pca.fit_transform(x_test)
# print(x_train)
# print(x_test)
# print(x_train.shape)
# model_ExtraTreeRegressor.fit(x_train,y_train)
# y_p = model_ExtraTreeRegressor.predict(x_test)
# mse = mean_squared_error(y_test,y_p)
# print(mse)
param_range= [5,6,7,8,8,9,10,11,12,13,14]
train_score,test_score = validation_curve(model_ExtraTreeRegressor,x_train,y_train,
                                          param_name='max_features',param_range=param_range,
                                          cv=10,scoring='neg_mean_squared_error')
train_score =  np.mean(train_score,axis=1)
test_score = np.mean(test_score,axis=1)
plt.plot(param_range,train_score,'o-',color = 'r',label = 'training')
plt.plot(param_range,test_score,'o-',color = 'g',label = 'testing')
plt.legend(loc='best')
plt.xlabel('iter')
plt.ylabel('mse')
plt.show()

