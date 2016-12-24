# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pylab as plt
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from xgboost.sklearn import XGBClassifier

from time_count import timer

@timer
def read_data():
    train_data = pd.read_csv('./data/tr_result_feature.csv')
    cv_data = pd.read_csv('./data/cv_result_feature.csv')
    return train_data, cv_data
    
@timer
def model_fit(estimator,X,y,use_train_cv=True,cv=5,early_stopping_rounds=50):
    if use_train_cv:
        xgb_params = estimator.get_xgb_params()
        train_data = xgb.DMatrix(data=X,
                                 label=y)
        cv_result = xgb.cv(xgb_params,
                           train_data,
                           num_boost_round=xgb_params['n_estimators'],
                           nfold=cv,
                           early_stopping_rounds=early_stopping_rounds,
                           metrics='auc',
                           show_progress=False)
        estimator.set_params(n_estimators=cv_result.shape[0])
    # 训练模型
    estimator.fit(X,y,eval_metric='auc')
    # 模型预测结果
    pred = estimator.predict(X)
    pred_prob = estimator.predict_proba(X)[:,1]
    # 打印模型信息
    print "Accuracy: %f." % accuracy_score(y,pred)
    print "AUC Score(Train): %f." % roc_auc_score(y,pred_prob)
    '''
    # 特征贡献排名
    feature_importance = pd.Series(estimator.booster().get_fscore()).sort_values(ascending=False)
    feature_importance.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    print feature_importance
    '''

    
        
    
    
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
    xgb_clf = XGBClassifier(learning_rate=0.1,
                            max_depth=5,
                            min_child_weight=1,
                            gamma=0,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            scale_pos_weight=1,
                            objective='binary:logistic'
                            )
    print xgb_clf.get_xgb_params()
    model_fit(xgb_clf,t_x,t_y)
    print xgb_clf.get_xgb_params()
    xgb_clf1 = XGBClassifier()
    print xgb_clf1.get_xgb_params()
    xgb_clf1 = xgb_clf1.set_params(**xgb_clf.get_xgb_params())
    print xgb_clf1.get_xgb_params()
    break