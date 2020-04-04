"""
###回归、分类概率-融合：

##简单加权平均，结果直接融合

#生成一些简单的样本数据，test_prei代表的是第i个模型的预测值
test_pre1 = [1.2,3.2,2.1,6.2]
test_pre2 = [0.9,3.1,2.0,5.9]
test_pre3 = [1.1,2.9,2.2,6.0]

#y_test_true 代表第i个模型的真实值
y_test_true = [1,3,2,6]

import numpy as np
import pandas as pd

#定义结果的加权平均函数
def weighted_method(test_pre1,test_pre2,test_pre3,w=[1/3,1/3,1/3]):
    weighted_result = w[0]*pd.Series(test_pre1)+w[1]*pd.Series(test_pre2)+w[2]*pd.Series(test_pre3)
    return weighted_result

from sklearn import metrics
#各模型的预测结果计算MAE
print('pred1 mae:',metrics.mean_absolute_error(y_test_true,test_pre1))
print('pred2 mae:',metrics.mean_absolute_error(y_test_true,test_pre2))
print('pred3 mae:',metrics.mean_absolute_error(y_test_true,test_pre3))

#根据加权计算MAE
w=[0.3,0.4,0.3]#定义比重权值
weighted_pre=weighted_method(test_pre1,test_pre2,test_pre3,w)
print('weighted_pre MAE:',metrics.mean_absolute_error(y_test_true,weighted_pre))

#可以发现加权结果相对于之前的结果是有提升的，这种我们称其为简单的加权平均。
#还有一些特殊的形式，比如mean平均，median平均

#定义结果的加权平均函数
def mean_method(test_pre1,test_pre2,test_pre3):
    mean_result = pd.concat([pd.Series(test_pre1),pd.Series(test_pre2),pd.Series(test_pre3)],axis=1).mean(axis=1)
    return mean_result

mean_pre=mean_method(test_pre1,test_pre2,test_pre3)
print('mean_pre mae:',metrics.mean_absolute_error(y_test_true,mean_pre))

#定义结果的加权平均函数
def median_method(test_pre1,test_pre2,test_pre3):
    median_result=pd.concat([pd.Series(test_pre1),pd.Series(test_pre2),pd.Series(test_pre3)],axis=1).median(axis=1)
    return median_result

median_pre=median_method(test_pre1,test_pre2,test_pre3)
print('median_pre MAE:',metrics.mean_absolute_error(y_test_true,median_pre))


## Stacking融合(回归)：
from sklearn import linear_model
def stacking_method(train_reg1,train_reg2,train_reg3,y_train_ture,test_pre1,test_pre2,test_pre3,model_L2=linear_model.LinearRegression()):
    model_L2.fit(pd.concat([pd.Series(train_reg1),pd.Series(train_reg2),pd.Series(train_reg3)],axis=1).values,y_train_ture)
    stacking_result=model_L2.predict(pd.concat([pd.Series(test_pre1),pd.Series(test_pre2),pd.Series(test_pre3)],axis=1).values)
    return stacking_result

## 生成一些简单的样本数据，test_prei 代表第i个模型的预测值
train_reg1 = [3.2, 8.2, 9.1, 5.2]
train_reg2 = [2.9, 8.1, 9.0, 4.9]
train_reg3 = [3.1, 7.9, 9.2, 5.0]
# y_test_true 代表第模型的真实值
y_train_true = [3, 8, 9, 5]

test_pre1 = [1.2, 3.2, 2.1, 6.2]
test_pre2 = [0.9, 3.1, 2.0, 5.9]
test_pre3 = [1.1, 2.9, 2.2, 6.0]

# y_test_true 代表第模型的真实值
y_test_true = [1, 3, 2, 6]

model_L2= linear_model.LinearRegression()
Stacking_pre = stacking_method(train_reg1,train_reg2,train_reg3,y_train_true,
                               test_pre1,test_pre2,test_pre3,model_L2)
print('Stacking_pre MAE:',metrics.mean_absolute_error(y_test_true, Stacking_pre))

##分类模型融合
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

##Voting投票机制：
#Voting即投票机制，分为软投票和硬投票两种，其原理采用少数服从多数的思想。

'''
硬投票：对多个模型直接进行投票，不区分模型结果的相对重要度，最终投票数最多的类为最终被预测的类。
'''
iris = datasets.load_iris()

x=iris.data
y=iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

clf1=XGBClassifier(learning_rate=0.1,n_estimators=150,max_depth=3,min_child_weight=2,subsample=0.7,
                   colsample_bytree=0.6,objective='binary:logistic')
clf2=RandomForestClassifier(n_estimators=50,max_depth=1,min_samples_split=4,
                            min_samples_leaf=63,oob_score=True)
clf3=SVC(C=0.1)

#硬投票
eclf=VotingClassifier(estimators=[('xgb','clf1'),('rf',clf2),('svc','clf3')],voting='hard')
for clf,label in zip([clf1,clf2,clf3,],['XGBBoosting','Random Forest','SVM','Ensemble']):
    scores = cross_val_score(clf,x,y,cv=5,scoring='accuracy')
    print('accurcy:%0.2f (+/- %0.2f)[%s]' % (scores.mean(),scores.std(),label))


'''
软投票：和硬投票原理相同，增加了设置权重的功能，可以为不同模型设置不同权重，进而区别模型不同的重要度。
'''
x=iris.data
y=iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

clf1=XGBClassifier(learning_rate=0.1,n_estimators=150,max_depth=3,min_child_weight=2,subsample=0.8,
                   colsample_bytree=0.8,objective='binary:logistic')
clf2=RandomForestClassifier(n_estimators=50,max_depth=1,min_samples_split=4,
                            min_samples_leaf=63,oob_score=True)
clf3=SVC(C=0.1,probability=True)

#软投票
eclf=VotingClassifier(estimators=[('xgb','clf1'),('rf','clf2'),('svc','clf3')],voting='soft',weights=[2,1,1])
clf1.fit(x_train,y_train)

for clf,label in zip([clf1,clf2,clf3,eclf],['XGBBoosting','Rabdom Forest','SVM','Ensemble']):
    scores = cross_val_score(clf,x,y,cv=5,scoring='accuracy')
    print("Accuracy:%0.2f (+/- %0.2f)[%s]" % (scores.mean(),scores.std(),label))


##分类的Stacking\Blending融合：
# #stacking是一种分层模型集成框架。
#  以两层为例，第一层由多个基学习器组成，其输入为原始训练集，
#  第二层的模型则是以第一层基学习器的输出作为训练集进行再训练，
#  从而得到完整的stacking模型, stacking两层模型都使用了全部的训练数据。
'''
5-fold stacking
'''
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier,GradientBoostingClassifier
import pandas as pd
#创建训练的数据集
data_0=iris.data
data=data_0[:100,:]

target_0 = iris.target
target = target_0[:100]

#模型融合中使用到各个单模型
clfs = [LogisticRegression(solver='lbfgs'),
        RandomForestClassifier(n_estimators=5,n_jobs=-1,criterion='gini'),
        ExtraTreesClassifier(n_estimators=5,n_jobs=-1,criterion='gini'),
        ExtraTreesClassifier(n_estimators=5,n_jobs=-1,criterion='entropy'),
        GradientBoostingClassifier(learning_rate=0.05,subsample=0.5,max_depth=6,n_estimators=5)]

#切分一部分数据作为测试集
X,X_predict,y,y_predict = train_test_split(data,target,test_size=0.3,random_state=2020)

dataset_blend_train = np.zeros((X.shape[0],len(clfs)))
dataset_blend_test = np.zeros((X_predict.shape[0],len(clfs)))

#5折stacking
n_splits = 5
skf = StratifiedKFold(n_splits)
skf = skf.split(X,y)

for j,clf in enumerate(clfs):
    # 依次训练各个单模型
    dataset_blend_test_j = np.zeros((X_predict.shape[0],5))
    for i,(train,test) in enumerate(skf):
        # 5-fold交叉训练，使用第i个部分作为预测， 剩余的部分来训练模型，获得其预测的输出作为第i部分的新特征。
        X_train,y_train,X_test,y_test = X[train],y[train],X[test],y[test]
        clf.fit(X_train,y_train)
        y_submission = clf.predict(X_test)[:,1]
        dataset_blend_train[test,j] = y_submission
        dataset_blend_test_j[:,i] = clf.predict_proba(X_predict)[:,1]
    #对于测试集，直接用这K个模型的预测值作为新的特征。
    dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)
    print("val auc acore:%f" % roc_auc_score(y_predict,dataset_blend_test[:,j]))

clf = LogisticRegression(solver='lbfgs')
clf.fit(dataset_blend_train,y)
y_submission = clf.predict_proba(dataset_blend_test)[:,1]

print("val auc score of stacking:%f" % (roc_auc_score(y_predict,y_submission)))

#Blending，其实和Stacking是一种类似的多层模型融合的形式
#
# 其主要思路是把原始的训练集先分成两部分，比如70%的数据作为新的训练集，剩下30%的数据作为测试集。
# 在第一层，我们在这70%的数据上训练多个模型，然后去预测那30%数据的label，同时也预测test集的label。
# 在第二层，我们就直接用这30%数据在第一层预测的结果做为新特征继续训练，然后用test集第一层预测的label做特征，用第二层训练的模型做进一步预测
#
## 其优点在于：
# 1.比stacking简单（因为不用进行k次的交叉验证来获得stacker feature）
# 2.避开了一个信息泄露问题：generlizers和stacker使用了不一样的数据集

## 缺点在于：
# 1.使用了很少的数据（第二阶段的blender只使用training set10%的量）
# 2.blender可能会过拟合
# 3.stacking使用多次的交叉验证会比较稳健 '''

'''Blending'''
#创建训练的数据集
data_0 = iris.data
data = data_0[:100,:]

target_0 = iris.target
target = target_0[:100]

#模型融合中使用到的各个单模型
clfs = [LogisticRegression(solver='lbfgs'),
        RandomForestClassifier(n_estimators=5,n_jobs=-1,criterion='gini'),
        RandomForestClassifier(n_estimators=5,n_jobs=-1,criterion='entropy'),
        ExtraTreesClassifier(n_estimators=5,n_jobs=-1,criterion='gini'),
        GradientBoostingClassifier(learning_rate=0.05,subsample=0.5,max_depth=6,n_estimators=5)]

#切分一部分数据作为测试集
X,X_predict,y,y_predict = train_test_split(data,target,test_size=0.3,random_state=2020)

#切分训练数据集为d1,d2两部分
X_d1,X_d2,y_d1,y_d2 = train_test_split(X,y,test_size=0.5,random_state=2020)
dataset_d1 = np.zeros((X_d2.shape[0],len(clfs)))
dataset_d2 = np.zeros((X_predict.shape[0],len(clfs)))

for j,clf in enumerate(clfs):
    #依次训练各个单模型
    clf.fit(X_d1,y_d1)
    y_submission = clf.predict_proba(X_d2)[:,1]
    dataset_d1[:,j] = y_submission
    #对于测试集，直接用这 K 个模型的预测值作为新的特征
    dataset_d2[:,j] = clf.predict_proba(X_predict)[:,1]
    print("val auc score:%f" % roc_auc_score(y_predict,dataset_d2[:,j]))

#融合使用的模型
clf = GradientBoostingClassifier(learning_rate=0.02,subsample=0.5,max_depth=6,n_estimators=30)
clf.fit(dataset_d1,y_d2)
y_submission= clf.predict_proba(dataset_d2)[:,1]
print("val auc score of blending:%f" % (roc_auc_score(y_predict,y_submission)))

"""

"""
##分类的Stacking融合(利用mlxtend)：
import warnings
warnings.filterwarnings('ignore')
import itertools
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.classifier import StackingClassifier

from sklearn.model_selection import cross_val_score
from mlxtend.plotting import plot_learning_curves
from mlxtend.plotting import plot_decision_regions

#以python自带的鸢尾花数据集为例
iris = datasets.load_iris()
X,y = iris.data[:,1:3],iris.target

clf1 = KNeighborsClassifier(n_neighbors=1)
clf2 = RandomForestClassifier(random_state=1)
clf2 = GaussianNB()
lr = LogisticRegression()
sclf = StackingClassifier(classifiers=[clf1,clf2,clf3],
                          meta_classifier=lr)

label = ['KNN','Random Forest','Naive Bayes','Stacking Classifier']
clf_list = [clf1,clf2,clf3,sclf]

fig = plt.figure(figsize=(10,8))

"""
#--------------------------------------------------------------------------------------------

#本赛题示例
import pandas as pd
import numpy as np
import warnings
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

import itertools
import matplotlib.gridspec as gridspec
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score,train_test_split

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from sklearn import linear_model
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.decomposition import PCA,FastICA,FactorAnalysis,SparsePCA

import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error,mean_squared_error


##数据读取
train_data = pd.read_csv('E:\\data\\二手车\\used_car_train_20200313.csv',sep=' ',engine='python')
testA_data = pd.read_csv('E:\\data\\二手车\\used_car_testA_20200313.csv',sep=' ',engine='python')
print(train_data.shape)
print(testA_data.shape)

numerical_cols = train_data.select_dtypes(exclude='object')._combine_match_columns
print(numerical_cols)

feature_cols = [col for col in numerical_cols if col not in ['SaleID','name','regDate','price']]

x_data = train_data[feature_cols]
y_data = train_data['price']
x_test = testA_data[feature_cols]
print('x train shape:',x_data.shape)
print('x test shape:',x_test.shape)

def sta_inf(data):
    print('_min',np.min(data))
    print('_max:', np.max(data))
    print('_mean', np.mean(data))
    print('_ptp', np.ptp(data))
    print('_std', np.std(data))
    print('_var', np.var(data))

print('sta of label:')
sta_inf(y_data)

x_data = x_data.fillna(-1)
x_test = x_test.fillna(-1)

def build_model_lr(x_train,y_train):
    reg_model = linear_model.LinearRegression()
    reg_model.fit(x_train,y_train)
    return reg_model

def build_model_ridge(x_train,y_train):
    reg_model = linear_model.Ridge(alpha=0.8)#alphas=range(1,100,5)
    reg_model.fit(x_train,y_train)
    return reg_model

def build_model_lasso(x_train,y_train):
    reg_model = linear_model.LassoCV()
    reg_model.fit(x_train,y_train)
    return reg_model

def build_model_gbdt(x_train,y_train):
    estimator =GradientBoostingRegressor(loss='ls',subsample= 0.85,max_depth= 5,n_estimators = 100)
    param_grid = {
            'learning_rate': [0.05,0.08,0.1,0.2],
            }
    gbdt = GridSearchCV(estimator, param_grid,cv=3)
    gbdt.fit(x_train,y_train)
    print(gbdt.best_params_)
    # print(gbdt.best_estimator_ )
    return gbdt

def build_model_xgb(x_train,y_train):
    model = xgb.XGBRegressor(n_estimators=120, learning_rate=0.08, gamma=0, subsample=0.8,\
        colsample_bytree=0.9, max_depth=5) #, objective ='reg:squarederror'
    model.fit(x_train, y_train)
    return model

def build_model_lgb(x_train,y_train):
    estimator = lgb.LGBMRegressor(num_leaves=63,n_estimators = 100)
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
    }
    gbm = GridSearchCV(estimator, param_grid)
    gbm.fit(x_train, y_train)
    return gbm


## xgb
xgr = xgb.XGBRegressor(n_estimators=120, learning_rate=0.1, subsample=0.8, \
                       colsample_bytree=0.9, max_depth=7)  # ,objective ='reg:squarederror'

scores_train = []
scores = []

## 5折交叉验证方式
sk = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
for train_ind, val_ind in sk.split(x_data, y_data):
    train_x = x_data.iloc[train_ind].values
    train_y = y_data.iloc[train_ind]
    val_x = x_data.iloc[val_ind].values
    val_y = y_data.iloc[val_ind]

    xgr.fit(train_x, train_y)
    pred_train_xgb = xgr.predict(train_x)
    pred_xgb = xgr.predict(val_x)

    score_train = mean_absolute_error(train_y, pred_train_xgb)
    scores_train.append(score_train)
    score = mean_absolute_error(val_y, pred_xgb)
    scores.append(score)

print('Train mae:', np.mean(score_train))
print('Val mae', np.mean(scores))


#划分数据集，并用多种方法训练和预测
## Split data with val
x_train,x_val,y_train,y_val = train_test_split(x_data,y_data,test_size=0.3)

## Train and Predict
print('Predict LR...')
model_lr = build_model_lr(x_train,y_train)
val_lr = model_lr.predict(x_val)
subA_lr = model_lr.predict(x_test)

print('Predict Ridge...')
model_ridge = build_model_ridge(x_train,y_train)
val_ridge = model_ridge.predict(x_val)
subA_ridge = model_ridge.predict(x_test)

print('Predict Lasso...')
model_lasso = build_model_lasso(x_train,y_train)
val_lasso = model_lasso.predict(x_val)
subA_lasso = model_lasso.predict(x_test)

print('Predict GBDT...')
model_gbdt = build_model_gbdt(x_train,y_train)
val_gbdt = model_gbdt.predict(x_val)
subA_gbdt = model_gbdt.predict(x_test)






#一般比赛中效果最为显著的两种方法
print('predict XGB...')
model_xgb = build_model_xgb(x_train,y_train)
val_xgb = model_xgb.predict(x_val)
subA_xgb = model_xgb.predict(x_test)

print('predict lgb...')
model_lgb = build_model_lgb(x_train,y_train)
val_lgb = model_lgb.predict(x_val)
subA_lgb = model_lgb.predict(x_test)

print('Sta inf of lgb:')
sta_inf(subA_lgb)



###加权融合
def Weighted_method(test_pre1,test_pre2,test_pre3,w=[1/3,1/3,1/3]):
    Weighted_result = w[0]*pd.Series(test_pre1)+w[1]*pd.Series(test_pre2)+w[2]*pd.Series(test_pre3)
    return Weighted_result

## Init the Weight
w = [0.3,0.4,0.3]

## 测试验证集准确度
val_pre = Weighted_method(val_lgb,val_xgb,val_gbdt,w)
MAE_Weighted = mean_absolute_error(y_val,val_pre)
print('MAE of Weighted of val:',MAE_Weighted)

## 预测数据部分
subA = Weighted_method(subA_lgb,subA_xgb,subA_gbdt,w)
print('Sta inf:')
sta_inf(subA)
## 生成提交文件
sub = pd.DataFrame()
sub['SaleID'] =x_test.index
sub['price'] = subA
sub.to_csv('./sub_Weighted.csv',index=False)

## 与简单的LR（线性回归）进行对比
val_lr_pred = model_lr.predict(x_val)
MAE_lr = mean_absolute_error(y_val,val_lr_pred)
print('MAE of lr:',MAE_lr)



##Starking融合
## Starking

## 第一层
train_lgb_pred = model_lgb.predict(x_train)
train_xgb_pred = model_xgb.predict(x_train)
train_gbdt_pred = model_gbdt.predict(x_train)

Strak_X_train = pd.DataFrame()
Strak_X_train['Method_1'] = train_lgb_pred
Strak_X_train['Method_2'] = train_xgb_pred
Strak_X_train['Method_3'] = train_gbdt_pred

Strak_X_val = pd.DataFrame()
Strak_X_val['Method_1'] = val_lgb
Strak_X_val['Method_2'] = val_xgb
Strak_X_val['Method_3'] = val_gbdt

Strak_X_test = pd.DataFrame()
Strak_X_test['Method_1'] = subA_lgb
Strak_X_test['Method_2'] = subA_xgb
Strak_X_test['Method_3'] = subA_gbdt


Strak_X_test.head()


## level2-method
model_lr_Stacking = build_model_lr(Strak_X_train,y_train)
## 训练集
train_pre_Stacking = model_lr_Stacking.predict(Strak_X_train)
print('MAE of Stacking-LR:',mean_absolute_error(y_train,train_pre_Stacking))

## 验证集
val_pre_Stacking = model_lr_Stacking.predict(Strak_X_val)
print('MAE of Stacking-LR:',mean_absolute_error(y_val,val_pre_Stacking))

## 预测集
print('Predict Stacking-LR...')
subA_Stacking = model_lr_Stacking.predict(Strak_X_test)
MAE of Stacking-LR: 628.399441036
MAE of Stacking-LR: 707.673951794
Predict Stacking-LR...
subA_Stacking[subA_Stacking<10]=10  ## 去除过小的预测值

sub = pd.DataFrame()
sub['SaleID'] = X_test.index
sub['price'] = subA_Stacking
sub.to_csv('./sub_Stacking.csv',index=False)
print('Sta inf:')
Sta_inf(subA_Stacking)







