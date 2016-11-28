import csv
import numpy as np
import sklearn
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