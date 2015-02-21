#!/usr/bin/python

import os

import matplotlib
matplotlib.use('Agg')
import pylab as pl
import numpy as np
import pandas as pd

from sklearn import cross_validation
from sklearn.ensemble import RandomForestRegressor

def load_data():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    ssub_df = pd.read_csv('sampleSubmission.csv')
    
    #for df in train_df, test_df, ssub_df:
        #print df.shape
        #print '\n%s' % '\n'.join(df.columns)
    
    #for c in train_df.columns:
        #print train_df[c].dtype, c
    
    #print train_df.describe()
    
    xtrain = train_df.drop(labels=['Id','Cover_Type'], axis=1).values
    ytrain = train_df['Cover_Type'].values
    xtest = test_df.drop(labels=['Id'], axis=1).values
    ytest = ssub_df['Id'].values
    
    print xtrain.shape, ytrain.shape, xtest.shape, ytest.shape
    print ytrain
    
    return xtrain, ytrain, xtest, ytest

def score_model(model, xtrain, ytrain):
    randint = reduce(lambda x,y: x|y, [ord(x)<<(n*8) for (n,x) in enumerate(os.urandom(4))])
    xTrain, xTest, yTrain, yTest = cross_validation.train_test_split(xtrain,
                                                                     ytrain,
                                                                     test_size=0.4, random_state=randint)
    model.fit(xTrain, yTrain)
    return model.score(xTest, yTest)

def prepare_submission(model, xtrain, ytrain, xtest, ytest):
    model.fit(xtrain, ytrain)
    ytest2 = model.predict(xtest).astype(np.int64)
    #dateobj = map(datetime.datetime.fromtimestamp, ytest)
    
    df = pd.DataFrame({'Id': ytest, 'Cover_Type': ytest2}, columns=('Id','Cover_Type'))
    df.to_csv('submission.csv', index=False)
    
    return


if __name__ == '__main__':
    xtrain, ytrain, xtest, ytest = load_data()

    model = RandomForestRegressor(n_estimators=400)
    #model = RandomForestRegressor()
    print 'score', score_model(model, xtrain, ytrain)
    print model.feature_importances_
    prepare_submission(model, xtrain, ytrain, xtest, ytest)
