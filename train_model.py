# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedShuffleSplit
from xgboost.sklearn import XGBClassifier

from time_count import timer

@timer
def read_data():
    train_data = pd.read_csv('./data/tr_result_feature.csv')
    cv_data = pd.read_csv('./data/cv_result_feature.csv')
    return train_data, cv_data
    
@timer
def model_fit(estimator,x,y,use_train_cv=True,cv=5,early_stopping_rounds=50):
    if use_train_cv:
        xgb_params = estimator.get_xgb_params()
        train_data = xgb.DMatrix(data=x,
                                 label=y)
        return type(train_data),xgb_params
        
    
    
# 读取数据
train_data, cv_data = read_data()

# 获取正负样本
train_positive = train_data[train_data['tag']==1]
train_negative = train_data[train_data['tag']==0]
cv_positive = cv_data[cv_data['tag']==1]
cv_negative = cv_data[cv_data['tag']==0]

# 

# 对负样本进行多次采样,使正负样本比例为1:19
sss = StratifiedShuffleSplit(n_splits=5,
                             train_size=len(train_positive)*19)
index_list = []
for train_index, test_index in sss.split(np.zeros(len(train_negative)),np.zeros(len(train_negative))):
    index_list.append(train_index)
    
for index in index_list:
    # 获得采样的 负样本 并与 正样本 拼接
    t_negative = train_negative.iloc[index]
    t_data = pd.concat((t_negative,train_positive),axis=0)
    t_data = t_data.sort_index(ascending=True)
    # 分离特征和标签
    t_x = t_data.iloc[:,3:]
    t_y = t_data.iloc[:,2]
    print model_fit(XGBClassifier(),t_x,t_y)
    break