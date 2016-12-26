# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pylab as plt
from sklearn.utils import shuffle
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from xgboost.sklearn import XGBClassifier

from time_count import timer

@timer
def read_data():
    train_data = pd.read_csv('./data/tr_result_feature.csv')
    cv_data = pd.read_csv('./data/cv_result_feature.csv')
    return train_data, cv_data
    
@timer
def model_fit(estimator,X,y,cv_X,cv_y,use_train_cv=True,cv=5,early_stopping_rounds=50):
    X = X.values
    y = y.values
    cv_X = cv_X.values
    cv_y = cv_y.values
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
                           show_progress=True)
        estimator.set_params(n_estimators=cv_result.shape[0])
    # 训练模型
    estimator.fit(X,y,eval_metric='auc')
    # 模型预测结果
    pred = estimator.predict(X)
    pred_prob = estimator.predict_proba(X)[:,1]
    pred_cv = estimator.predict(cv_X)
    pred_prob_cv = estimator.predict_proba(cv_X)[:,1]
    # 打印模型信息
    print "Accuracy: %f." % accuracy_score(y,pred)
    print "AUC Score(Train): %f." % roc_auc_score(y,pred_prob)
    print "AUC Score(CV): %f." % roc_auc_score(cv_y,pred_prob_cv)
    '''
    # 特征贡献排名
    feature_importance = pd.Series(estimator.booster().get_fscore()).sort_values(ascending=False)
    feature_importance.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    print feature_importance
    '''
    
@timer
def grid_model_predict(estimator,X,y,cv_X,cv_y):
    X = X.values
    y = y.values
    cv_X = cv_X.values
    cv_y = cv_y.values
    pred = estimator.predict(X)
    pred_prob = estimator.predict_proba(X)[:,1]
    pred_cv = estimator.predict(cv_X)
    pred_prob_cv = estimator.predict_proba(cv_X)[:,1]
    print "Accuracy: %f." % accuracy_score(y,pred)
    print "AUC Score(Train): %f." % roc_auc_score(y,pred_prob)
    print "AUC Score(CV): %f." % roc_auc_score(cv_y,pred_prob_cv)    
    
@timer
def grid_fit(estimator,params,X,y):
    grid_model = GridSearchCV(estimator=estimator,
                              param_grid=params,
                              scoring='roc_auc',
                              iid=True,
                              cv=5)
    X = X.values
    y = y.values
    grid_model.fit(X,y)
    print grid_model.grid_scores_
    print grid_model.best_score_
    print grid_model.best_params_
    return grid_model
    

# 读取数据
train_data, cv_data = read_data()

# 获取正负样本
train_positive = train_data[train_data['tag']==1]
train_negative = train_data[train_data['tag']==0]
cv_positive = cv_data[cv_data['tag']==1]
cv_negative = cv_data[cv_data['tag']==0]
cv_x = cv_data.iloc[:,3:]
cv_y = cv_data.iloc[:,2]

# 对负样本进行多次采样,使正负样本比例为1:19
sss = StratifiedShuffleSplit(n_splits=5,
                             train_size=len(train_positive)*19,
                             random_state=7)
index_list = []
for train_index, test_index in sss.split(np.zeros(len(train_negative)),np.zeros(len(train_negative))):
    index_list.append(train_index)

data_list = []
for index in index_list:
    # 获得采样的 负样本 并与 正样本 拼接
    t_negative = train_negative.iloc[index]
    t_data = pd.concat((t_negative,train_positive),axis=0)
    t_data = t_data.sort_index(ascending=True)
    data_list.append(t_data)

    

# 开始训练多个模型



# 第一个
t_data = data_list[0]
# 分离特征和标签
t_x = t_data.iloc[:,3:]
t_y = t_data.iloc[:,2]
# 初始化设置
xgb_clf = XGBClassifier(learning_rate=0.1,
                        n_estimators=500,
                        max_depth=5,
                        min_child_weight=1,
                        gamma=0,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        scale_pos_weight=1,
                        objective='binary:logistic'
                        )
# 评估需要的树的数量
#model_fit(xgb_clf,t_x,t_y,cv_x,cv_y)


## 参数调优:max_depth, min_child_weight
#xgb_clf1 = XGBClassifier().set_params(**xgb_clf.get_xgb_params())
#params = {'max_depth':[3],
#          'min_child_weight':[1]}
#grid_model = grid_fit(xgb_clf1,params,t_x,t_y)
#xgb_clf1.set_params(**grid_model.best_params_)  # 设置参数
#advanced_params = xgb_clf1.get_xgb_params()
#grid_model_predict(grid_model,t_x,t_y,cv_x,cv_y) # 评测结果(CV)


## 参数调优:gamma
#params = {'gamma':[0.6]}
#xgb_clf2 = XGBClassifier(learning_rate=0.1,
#                         max_depth=3,
#                         min_child_weight=1,
#                         subsample=0.8,
#                         colsample_bytree=0.8,
#                         scale_pos_weight=1,
#                         objective='binary:logistic')
#grid_model = grid_fit(xgb_clf2,params,t_x,t_y)
#xgb_clf2.set_params(**grid_model.best_params_)
#advanced_params = xgb_clf2.get_xgb_params()
#grid_model_predict(grid_model,t_x,t_y,cv_x,cv_y) # 评测结果(CV):0.915631


## 参数调优:subsample, colsample_bytree
#params = {'subsample':[0.8],
#          'colsample_bytree':[0.8]}
#xgb_clf3 = XGBClassifier(learning_rate=0.1,
#                         max_depth=3,
#                         min_child_weight=1,
#                         gamma=0.6,
#                         subsample=0.8,
#                         colsample_bytree=0.8,
#                         scale_pos_weight=1,
#                         objective='binary:logistic')
#grid_model = grid_fit(xgb_clf3,params,t_x,t_y)
#xgb_clf3.set_params(**grid_model.best_params_)
#advanced_params = xgb_clf3.get_xgb_params()
#grid_model_predict(grid_model,t_x,t_y,cv_x,cv_y) # 评测结果(CV):0.915631

## 参数调优:reg_alpha
#params = {'reg_alpha':[0.01]}
#xgb_clf4 = XGBClassifier(learning_rate=0.1,
#                         max_depth=3,
#                         min_child_weight=1,
#                         gamma=0.6,
#                         subsample=0.8,
#                         colsample_bytree=0.8,
#                         scale_pos_weight=1,
#                         objective='binary:logistic')
#grid_model = grid_fit(xgb_clf4,params,t_x,t_y)
#xgb_clf4.set_params(**grid_model.best_params_)
#advanced_params = xgb_clf4.get_xgb_params()
#grid_model_predict(grid_model,t_x,t_y,cv_x,cv_y) # 评测结果(CV):0.916113

## 参数调优:scale_pos_weight
#params = {'scale_pos_weight':[1]}
#xgb_clf5 = XGBClassifier(learning_rate=0.1,
#                         max_depth=3,
#                         min_child_weight=1,
#                         gamma=0.6,
#                         subsample=0.8,
#                         colsample_bytree=0.8,
#                         reg_alpha=0.01,
#                         scale_pos_weight=1,
#                         objective='binary:logistic')
#grid_model = grid_fit(xgb_clf5,params,t_x,t_y)
#xgb_clf5.set_params(**grid_model.best_params_)
#advanced_params = xgb_clf5.get_xgb_params()
#grid_model_predict(grid_model,t_x,t_y,cv_x,cv_y) # 评测结果(CV):0.916113

## 参数调优:learning_rate
#xgb_clf5 = XGBClassifier(learning_rate=0.01,
#                         n_estimators=5000,
#                         max_depth=3,
#                         min_child_weight=1,
#                         gamma=0.6,
#                         subsample=0.8,
#                         colsample_bytree=0.8,
#                         reg_alpha=0.01,
#                         scale_pos_weight=1,
#                         objective='binary:logistic')
#model_fit(xgb_clf5,t_x,t_y,cv_x,cv_y)# 评测结果(CV):0.917425

# 模型保存
xgb_clf_1 = XGBClassifier(learning_rate=0.01,
                         n_estimators=2050,
                         max_depth=3,
                         min_child_weight=1,
                         gamma=0.6,
                         subsample=0.8,
                         colsample_bytree=0.8,
                         reg_alpha=0.01,
                         scale_pos_weight=1,
                         objective='binary:logistic')
xgb_clf_1.fit(t_x.values,t_y.values)
joblib.dump(xgb_clf_1,'1.pkl')
xgb_clf_1 = joblib.load('1.pkl')

pred = xgb_clf_1.predict(t_x.values)
pred_prob = xgb_clf_1.predict_proba(t_x.values)[:,1]
pred_cv = xgb_clf_1.predict(cv_x.values)
pred_prob_cv = xgb_clf_1.predict_proba(cv_x.values)[:,1]

print "Accuracy: %f." % accuracy_score(t_y.values,pred)
print "AUC Score(Train): %f." % roc_auc_score(t_y.values,pred_prob)
print "AUC Score(CV): %f." % roc_auc_score(cv_y.values,pred_prob_cv)