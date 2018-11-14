import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore') #抑制warning输出
#读取数据
x = pd.read_csv("train.csv")
# x = x[:100]
# print(x.columns)
# for i in x['房屋朝向']:
#     print(i)
# print(len(x['房屋朝向'].unique()))
# for i in enumerate(x['房屋朝向'].unique()):
#     print(i,end="\t")
#数据预处理
#包含缺失值的处理，对一些离散特征进行独热编码，特征选择（降维）？，特征缩放，数据分割等
#x.pop('id') #去掉id列,id没什么用，，，，训练集没有id，测试集才有，，，
#特征映射，将朝向转化成编码
# trans_mapping = {"东":1,"南":2,"西":3,"北":4,     
#                  "东北":5,"东南":6,"西北":7,"西南":8} #然后要使用独热编码转换
# x['房屋朝向'] = x['房屋朝向'].map(trans_mapping)

# lb = LabelEncoder()
# x['房屋朝向'] = lb .fit_transform(x['房屋朝向'])
# oneHot = pd.get_dummies(x['房屋朝向'])#直接进行了独热编码，比较简洁
# print(oneHot)
# x.pop("房屋朝向")
# x.join(oneHot)
# ohe = OneHotEncoder(categorical_features=)
print(x.shape)
x_train = x.iloc[:,:18]
y_train = x.iloc[:,18]
x_train=pd.get_dummies(x_train) #独热编码
#缺失值处理

x_train.pop('居住状态') #缺少89.75%的数据，直接弹出，没什么用
x_train.pop('出租方式') #同理弹出
x_train.pop('装修情况')

#区缺失较少，使用最频繁策略;小区房屋出租数量变化大，缺失1001个，使用median策略
values={'区':12.0,
        '小区房屋出租数量':0.08203125,
        '位置':52,                         #位置采取的是最频繁策略
        '地铁线路':0,                      #地铁线路为nan的时候，地铁站点、距离都是nan所以认为是没有
        '地铁站点':0,                      #地铁附近，所以全部设置为0
        '距离':0}
# print(x.columns)
x_train=x_train.fillna(value=values)#分别对应处理缺失值
# print("shape: {}, columns: {}".format(x.shape,x.columns))
# print(x.head)
# imp = SimpleImputer(missing_values = np.nan,
#               strategy ='mean') #sklearn提供mean，medium,most_frequent等策略
#数据分割
# x_train,x_test,y_train,y_test = train_test_split(x.iloc[:,:18],x.iloc[:,18],test_size=0.01,random_state=0)

#特征值缩放
stdsc = StandardScaler()
x_train_std = stdsc.fit_transform(x_train)
# x_test_std =  stdsc.fit_transform()
num = int(x_train_std.shape[0]*0.8)
temp = x_train_std
x_train_std = temp[:num,:]
x_test =  temp[num:,:]
temp = y_train
y_train =temp[:num]
y_test = temp[num:]

def prepare(path):
    data = pd.read_csv("test.csv")
    if 'id' in data.columns: 
            data.pop('id')
    data.pop('居住状态') #缺少89.75%的数据，直接弹出，没什么用
    data.pop('出租方式') #同理弹出
    data.pop('装修情况')
    data = pd.get_dummies(data)
    print(data.shape)
    values={'区':12.0,
        '小区房屋出租数量':0.08203125,
        '位置':52,                         #位置采取的是最频繁策略
        '地铁线路':0,                      #地铁线路为nan的时候，地铁站点、距离都是nan所以认为是没有
        '地铁站点':0,                      #地铁附近，所以全部设置为0
        '距离':0}
    data=data.fillna(value=values)#分别对应处理缺失值
    stdsc = StandardScaler()
    return  stdsc.fit_transform(data)

# elas.fit(x_train_std,y_train)
# y_pred = elas.predict(x_test_std)
# result = mean_squared_error(y_test,y_pred)
# print("mse:  {}".format(result))
#嵌套交叉验证选择算法,待选择算法：拉索回归，岭回归，弹性网络，多项回归
args = [('scl',StandardScaler()),
        ('pca',PCA(n_components=10)), #要不要pca降维存疑
        ("regression",ElasticNet())]
param_range = [1.0,2,3,4,5]
ratio = [0.05,0.07,0.1,0.3,0.5,0.6,0.7,0.9]  #list(np.arange(50)/1000) #
boolean= [False,True]
pipe = Pipeline([('stdsc',stdsc),
                 ('pca',PCA(n_components=10)),
                 ('r',ElasticNet(alpha=1,normalize=False,random_state=1,tol=1,fit_intercept=True,max_iter=200))])
params = [{#"alpha":param_range,
               "l1_ratio":[0.01,0.09,0.1,0.3,0.5,0.6,0.7,0.9,0.05],
             #  "normalize":False,
            #  "tol":10,  #The tolerance for the optimization:loss?
             #  "positive":boolean,
          #     "max_iter":[100,200,150,300]
               },]

# gs = GridSearchCV(estimator=elas,
#                   param_grid=[{'l1_ratio':ratio}],
#                   scoring="neg_mean_squared_error",
#                   cv = 10)
# gs = gs.fit(x_train,y_train)
# print(gs.best_score_)
# print(gs.best_params_)
#输出

pca = PCA(n_components=30)
x_test = pca.fit_transform(x_test)
x_train_std = pca.fit_transform(x_train_std)

elas = ElasticNet(l1_ratio=0.4,normalize=False,max_iter=100,tol=10,positive=False,
                   random_state=0)  #alpha 默认为1，所以a=0.4，b=0.6  7 0.4 
  
elas.fit(x_train_std,y_train)      

y_p = elas.predict(x_test)
mse = mean_squared_error(y_test,y_p)
print("mse is : {}".format(mse))

data = prepare('test.csv')
data = pca.fit_transform(data)
y_p = elas.predict(data)
df = pd.DataFrame({'id':list(range(1,y_p.shape[0]+1)),'price':y_p})
df.to_csv('final.csv',index=0)
print(y_p)
# train_score,test_score = validation_curve(elas,x_train_std,y_train,param_name='positive',param_range=param_range,cv=10,scoring='neg_mean_squared_error')
# train_score =  np.mean(train_score,axis=1)
# test_score = np.mean(test_score,axis=1)
# plt.plot(param_range,train_score,'o-',color = 'r',label = 'training')
# plt.plot(param_range,test_score,'o-',color = 'g',label = 'testing')
# plt.legend(loc='best')
# plt.xlabel('iter')
# plt.ylabel('mse')
# plt.show()

# elas.fit(x_train_std,y_train)


# train_sizes,train_score,test_score = learning_curve(elas,x_test_std,y_test,train_sizes=[0.1,0.2,0.4,0.6,0.8,1],cv=10,scoring='neg_mean_squared_error')
# train_error =  1- np.mean(train_score,axis=1)
# test_error = 1- np.mean(test_score,axis=1)
# plt.plot(train_sizes,train_error,'o-',color = 'r',label = 'training')
# plt.plot(train_sizes,test_error,'o-',color = 'g',label = 'testing')
# plt.legend(loc='best')
# plt.xlabel('traing examples')
# plt.ylabel('error')
# plt.show()