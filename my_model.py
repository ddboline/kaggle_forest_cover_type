#!/usr/bin/python

import os

import matplotlib
matplotlib.use('Agg')
import pylab as pl
import numpy as np
import pandas as pd

from sklearn import cross_validation
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.cluster import MiniBatchKMeans
from sklearn.neural_network import BernoulliRBM

from sklearn.pipeline import Pipeline

from sklearn.externals import joblib

from sklearn.metrics import accuracy_score

def gaussian(x, mu, sig):
    return np.exp(-(x-mu)**2/(2*sig**2))/(sig*np.sqrt(2*np.pi))

def fit_func(x, *p):
    return p[2] * gaussian(x, p[0], p[1])

def get_plots(in_df):
    print in_df.columns
    
    for c in in_df.columns:
        if c in ('Id', 'Cover_Type'):
            continue
        pl.clf()
        nent = len(in_df[c])
        hmin, hmax = in_df[c].min(), in_df[c].max()
        xbins = np.linspace(hmin,hmax,nent//500)
        for n in range(1,8):
            covtype = in_df.Cover_Type == n
            a = in_df[covtype][c].values
            #b = in_df[covtype][c].hist(bins=xbins, histtype='step')
            pl.hist(a, bins=xbins, histtype='step')
            if c == 'Elevation':
                mu, sig = a.mean(), a.std()
                x = np.linspace(hmin,hmax,1000)
                y = (a.sum()/len(xbins)) * gaussian(x, mu, sig)
                pl.plot(x, y, '--')
        pl.title(c)
        pl.savefig('%s.png' % c)

def make_hash(in_df):
    tmp = in_df['Soil_Type1'].values
    for n in range(2,40+1):
        tmp2 = in_df['Soil_Type%s' % n].values
        tmp |= (tmp2 << (n-1))
    in_df['Soil_Type_Hash'] = tmp
    cols = []
    for n in range(1,40+1):
        cols.append('Soil_Type%s' % n)
    in_df = in_df.drop(labels=cols, axis=1)
    return in_df

def load_data():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    ssub_df = pd.read_csv('sampleSubmission.csv')
    
    #get_plots(train_df)

    #xtrain = train_df.drop(labels=['Id','Cover_Type'], axis=1).values
    #ytrain = train_df['Cover_Type'].values
    #xtest = test_df.drop(labels=['Id'], axis=1).values
    
    #model = KNeighborsClassifier(7)
    #model.fit(xtrain, ytrain)
    #train_df['knn'] = model.predict(xtrain)
    #test_df['knn'] = model.predict(xtest)

    xtrain = train_df.drop(labels=['Id','Cover_Type'], axis=1).values
    ytrain = train_df['Cover_Type'].values
    xtest = test_df.drop(labels=['Id'], axis=1).values
    ytest = ssub_df['Id'].values
    
    print xtrain.shape, ytrain.shape, xtest.shape, ytest.shape
    
    return xtrain, ytrain, xtest, ytest

def score_model(model, xtrain, ytrain):
    randint = reduce(lambda x,y: x|y, [ord(x)<<(n*8) for (n,x) in enumerate(os.urandom(4))])
    xTrain, xTest, yTrain, yTest = cross_validation.train_test_split(xtrain,
                                                                     ytrain,
                                                                     test_size=0.4, random_state=randint)
    model.fit(xTrain, yTrain)
    ytest_pred = model.predict(xTest)
    print 'accuracy', accuracy_score(ytest_pred,yTest)
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

    #pca = PCA()
    #pca.fit(xtrain)
    
    #xtrain = pca.transform(xtrain)
    #xtest = pca.transform(xtest)

    #model = RandomForestRegressor(n_jobs=-1)
    #model = RandomForestClassifier(n_estimators=400)
    #model = GradientBoostingClassifier()
    #model = KNeighborsClassifier(7)
    
    model = Pipeline([('pca', PCA()),
                      ('knn', KMeans(7)),
                      ('rf', RandomForestClassifier(n_estimators=400)),])
    
    print 'score', score_model(model, xtrain, ytrain)
    #print model.feature_importances_
    prepare_submission(model, xtrain, ytrain, xtest, ytest)
