import csv
import numpy as np
import sklearn
import sklearn.preprocessing
from sklearn.linear_model import LogisticRegression
import sklearn.pipeline
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
import seaborn as sns
import copy
import datetime
import itertools
import os, sys
sys.path.append('/home/mbocamazo/MLCC/results')

from PIL import Image
sns.set_style("darkgrid", {"grid.linewidth": .5, "axes.facecolor": ".9"})
deep2 = [(0.2980392156862745, 0.4470588235294118, 0.6901960784313725),
 (0.3333333333333333, 0.6588235294117647, 0.40784313725490196),
 (0.7686274509803922, 0.3058823529411765, 0.3215686274509804),
 (0.8, 0.7254901960784313, 0.4549019607843137),
 (0.39215686274509803, 0.7098039215686275, 0.803921568627451),
 (0.5058823529411764, 0.4470588235294118, 0.6980392156862745)]
sns.set_palette(deep2)

def build_dict(xs):
    d = {}
    for x in xs:
        if not(x in d.keys()):
            d[x] = []
    return d

def tt_split(X,y,split):    
    Xtrain = X.iloc[:split,:]
    Xtest = X.iloc[split:,:]
    ytrain = y.iloc[:split]
    ytest = y.iloc[split:]
    return Xtrain, Xtest, ytrain, ytest

def print_predict(X_ml, model, ytrain, ytest, split):
    py_x = model.predict_proba(X_ml)
    dy_x = model.predict(X_ml)
    print("Training Log Loss: " + str(sklearn.metrics.log_loss(ytrain, py_x[:split])))
    print("Training Accuracy: " + str(sklearn.metrics.accuracy_score(ytrain, dy_x[:split])))
    print("Testing  Log Loss: " + str(sklearn.metrics.log_loss(ytest, py_x[split:])))
    print("Testing  Accuracy: " + str(sklearn.metrics.accuracy_score(ytest, dy_x[split:])))

def construct_label_vec(X, dy_x1, dy_x2):
    pointer_1 = 0
    pointer_2 = 0
    pointer_label = 0
    label_vec = np.zeros((len(X),1))
    for i in X.index:
        if X.loc[i,'CAR_AV'] == 1:
            if dy_x1[pointer_1]==1:
                label_vec[pointer_label] = 3 # this is the code for CAR
            else:
                label_vec[pointer_label] = dy_x2[pointer_2]
                pointer_2 += 1
            pointer_1 += 1
        else:
            label_vec[pointer_label] = dy_x2[pointer_2]
            pointer_2 += 1
        pointer_label +=1 
    return label_vec
            

def multistage_model(Xtrain, Xtest, ytrain, ytest, ML_feat):
    # TRAIN
    Xtrain_car_av_index = Xtrain['CAR_AV']==1
    Xtrain_car_av = Xtrain[Xtrain_car_av_index]
    # this indexes into the CAR_AV examples and checks if the chosen mode is CAR
    ytrain_car_non = ytrain[Xtrain_car_av_index] == 3 

    scaler = sklearn.preprocessing.StandardScaler()
    clf = LogisticRegression()
    model1 = sklearn.pipeline.Pipeline([('scaler',scaler),('LogReg',clf)])
    model1.fit(Xtrain_car_av[ML_feat], ytrain_car_non)
    dy_x = model1.predict(Xtrain_car_av[ML_feat])
    # these are the discrete predictions of car presence
    # so we take the negative predictions as part of the training data for next stage
    Xtrain_pTR = Xtrain_car_av[~dy_x]
    ytrain_car_av = ytrain[Xtrain['CAR_AV']==1]
    ytrain_pTR = ytrain_car_av[~dy_x]
    Xtrain_no_car = Xtrain[~Xtrain_car_av_index]
    ytrain_no_car = ytrain[~Xtrain_car_av_index]
    Xtr_stage2 = pd.concat([Xtrain_pTR, Xtrain_no_car])
    ytr_stage2 = pd.concat([ytrain_pTR, ytrain_no_car])

    scaler2 = sklearn.preprocessing.StandardScaler()
    clf2 = LogisticRegression()
    model2 = sklearn.pipeline.Pipeline([('scaler',scaler2),('LogReg',clf2)])
    model2.fit(Xtr_stage2[ML_feat], ytr_stage2)
    dy_x2 = model2.predict(Xtr_stage2[ML_feat])
    dy_train = construct_label_vec(Xtrain, dy_x, dy_x2)
    train_acc = sklearn.metrics.accuracy_score(ytrain, dy_train)

    # TEST
    Xtest_car_av_index = Xtest['CAR_AV']==1
    Xtest_car_av = Xtest[Xtest_car_av_index]    
    ytest_car_non = ytest[Xtest_car_av_index]==3 # this indexes into the CAR_AV examples
    dy_x_test = model1.predict(Xtest_car_av[ML_feat])
    
    ytest_car_av = ytest[Xtest_car_av_index]
    Xtest_pTR = Xtest_car_av[~dy_x_test]
    ytest_pTR = ytest_car_av[~dy_x_test]
    Xtest_no_car = Xtest[Xtest['CAR_AV']!=1]
    ytest_no_car = ytest[Xtest['CAR_AV']!=1]

    Xte_stage2 = pd.concat([Xtest_pTR, Xtest_no_car])
    yte_stage2 = pd.concat([ytest_pTR, ytest_no_car])

    dy_x2_test = model2.predict(Xte_stage2[ML_feat])
    dy_test = construct_label_vec(Xtest, dy_x_test, dy_x2_test)
    test_acc = sklearn.metrics.accuracy_score(ytest, dy_test)

    return model1, model2, dy_train, dy_test


def multistage_model_gen(Xtrain, Xtest, ytrain, ytest, ML_feat, clf1, clf2):
    # TRAIN
    Xtrain_car_av_index = Xtrain['CAR_AV']==1
    Xtrain_car_av = Xtrain[Xtrain_car_av_index]
    # this indexes into the CAR_AV examples and checks if the chosen mode is CAR
    ytrain_car_non = ytrain[Xtrain_car_av_index] == 3 

    scaler = sklearn.preprocessing.StandardScaler()
    model1 = sklearn.pipeline.Pipeline([('scaler',scaler),('Clf1',clf1)])
    model1.fit(Xtrain_car_av[ML_feat], ytrain_car_non)
    dy_x = model1.predict(Xtrain_car_av[ML_feat])
    # these are the discrete predictions of car presence
    # so we take the negative predictions as part of the training data for next stage
    Xtrain_pTR = Xtrain_car_av[~dy_x]
    ytrain_car_av = ytrain[Xtrain['CAR_AV']==1]
    ytrain_pTR = ytrain_car_av[~dy_x]
    Xtrain_no_car = Xtrain[~Xtrain_car_av_index]
    ytrain_no_car = ytrain[~Xtrain_car_av_index]
    Xtr_stage2 = pd.concat([Xtrain_pTR, Xtrain_no_car])
    ytr_stage2 = pd.concat([ytrain_pTR, ytrain_no_car])

    scaler2 = sklearn.preprocessing.StandardScaler()
    model2 = sklearn.pipeline.Pipeline([('scaler',scaler2),('Clf2',clf2)])
    model2.fit(Xtr_stage2[ML_feat], ytr_stage2)
    dy_x2 = model2.predict(Xtr_stage2[ML_feat])
    dy_train = construct_label_vec(Xtrain, dy_x, dy_x2)
    train_acc = sklearn.metrics.accuracy_score(ytrain, dy_train)

    # TEST
    Xtest_car_av_index = Xtest['CAR_AV']==1
    Xtest_car_av = Xtest[Xtest_car_av_index]    
    ytest_car_non = ytest[Xtest_car_av_index]==3 # this indexes into the CAR_AV examples
    dy_x_test = model1.predict(Xtest_car_av[ML_feat])
    
    ytest_car_av = ytest[Xtest_car_av_index]
    Xtest_pTR = Xtest_car_av[~dy_x_test]
    ytest_pTR = ytest_car_av[~dy_x_test]
    Xtest_no_car = Xtest[Xtest['CAR_AV']!=1]
    ytest_no_car = ytest[Xtest['CAR_AV']!=1]

    Xte_stage2 = pd.concat([Xtest_pTR, Xtest_no_car])
    yte_stage2 = pd.concat([ytest_pTR, ytest_no_car])

    dy_x2_test = model2.predict(Xte_stage2[ML_feat])
    dy_test = construct_label_vec(Xtest, dy_x_test, dy_x2_test)
    test_acc = sklearn.metrics.accuracy_score(ytest, dy_test)

    return model1, model2, dy_train, dy_test

# from http://zacstewart.com/2014/08/05/pipelines-of-featureunions-of-pipelines.html
"""
class ColumnExtractor(TransformerMixin):

    def __init__(self, columns=[]):
        self.columns = columns

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def transform(self, X, **transform_params):
        return X[self.columns]

    def fit(self, X, y=None, **fit_params):
        return self

class HourOfDayTransformer(TransformerMixin):

    def transform(self, X, **transform_params):
        hours = DataFrame(X['datetime'].apply(lambda x: x.hour))
        return hours

    def fit(self, X, y=None, **fit_params):
        return self

class ModelTransformer(TransformerMixin):

    def __init__(self, model):
        self.model = model

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)
        return self

    def transform(self, X, **transform_params):
        return DataFrame(self.model.predict(X))
"""