#!/usr/bin/python

import os

import matplotlib
matplotlib.use('Agg')
import pylab as pl
import numpy as np
import pandas as pd

from sklearn import cross_validation

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.cluster import MiniBatchKMeans
from sklearn.neural_network import BernoulliRBM

from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier

from sklearn.decomposition import PCA, FastICA, KernelPCA

from sklearn.pipeline import Pipeline

from sklearn.externals import joblib

from sklearn.metrics import accuracy_score

def gaussian(x, mu, sig):
    return np.exp(-(x-mu)**2/(2*sig**2))/(sig*np.sqrt(2*np.pi))

def fit_func(x, *p):
    return p[2] * gaussian(x, p[0], p[1])

def create_html_page_of_plots(list_of_plots):
    if not os.path.exists('html'):
        os.makedirs('html')
    os.system('mv *.png html')
    print(list_of_plots)
    with open('html/index.html', 'w') as htmlfile:
        htmlfile.write('<!DOCTYPE html><html><body><div>')
        for plot in list_of_plots:
            htmlfile.write('<p><img src="%s"></p>' % plot)
        htmlfile.write('</div></html></html>')

def get_plots(in_df):
    list_of_plots = []
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
            #if c == 'Elevation':
                #mu, sig = a.mean(), a.std()
                #x = np.linspace(hmin,hmax,1000)
                #y = (a.sum()/len(xbins)) * gaussian(x, mu, sig)
                #pl.plot(x, y, '--')
        pl.title(c)
        pl.savefig('%s.png' % c)
        list_of_plots.append('%s.png' % c)
    create_html_page_of_plots(list_of_plots)
    
def plot_failures(in_array, covertype):
    print in_array.shape
    list_of_plots = []
    
    for c in range(in_array.shape[1]):
        pl.clf()
        nent = in_array.shape[0]
        hmin, hmax = in_array[:,c].min(), in_array[:,c].max()
        xbins = np.linspace(hmin,hmax,20)
        for n in range(1,8):
            covtype = covertype == n
            a = in_array[covtype][:,c]
            pl.hist(a, bins=xbins, histtype='step')
        pl.title(c)
        pl.savefig('%s.png' % c)
        list_of_plots.append('%s.png' % c)
    create_html_page_of_plots(list_of_plots)
        
    
def transform_from_classes(inp):
    y = np.zeros((inp.shape[0], 7), dtype=np.int64)
    for (index, Class) in enumerate(inp):
        cidx = Class-1
        y[index, cidx] = 1.0
    return y

def transform_to_class(yinp):
    return np.array(map(lambda x: x+1, np.argmax(yinp, axis=1)))


def load_data():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    ssub_df = pd.read_csv('sampleSubmission.csv')
    
    get_plots(train_df)

    #xtrain = train_df.drop(labels=['Id','Cover_Type'], axis=1).values
    #ytrain = train_df['Cover_Type'].values
    #xtest = test_df.drop(labels=['Id'], axis=1).values
    
    #model = KNeighborsClassifier(7)
    #model.fit(xtrain, ytrain)
    #train_df['knn'] = model.predict(xtrain)
    #test_df['knn'] = model.predict(xtest)

    #labels_to_drop = []
                      #'Soil_Type5', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9',
                      #'Soil_Type15', 'Soil_Type19', 'Soil_Type21',
                      #'Soil_Type25', 'Soil_Type26', 'Soil_Type27',
                      #'Soil_Type28', 'Soil_Type34', 'Soil_Type36',
                      #]
    labels_to_drop = [
        #u'Elevation', 
        #u'Aspect', 
        #u'Slope', 
        #u'Horizontal_Distance_To_Hydrology', 
        #u'Vertical_Distance_To_Hydrology', 
        #u'Horizontal_Distance_To_Roadways', 
        #u'Hillshade_9am', 
        #u'Hillshade_Noon', 
        #u'Hillshade_3pm', 
        #u'Horizontal_Distance_To_Fire_Points', 
        #u'Wilderness_Area1', 
        #u'Wilderness_Area2', 
        #u'Wilderness_Area3', 
        #u'Wilderness_Area4', 
        #u'Soil_Type1', u'Soil_Type2', u'Soil_Type3', u'Soil_Type4', u'Soil_Type5', u'Soil_Type6', u'Soil_Type7', u'Soil_Type8', u'Soil_Type9', u'Soil_Type10', u'Soil_Type11', u'Soil_Type12', u'Soil_Type13', u'Soil_Type14', u'Soil_Type15', u'Soil_Type16', u'Soil_Type17', u'Soil_Type18', u'Soil_Type19', u'Soil_Type20', u'Soil_Type21', u'Soil_Type22', u'Soil_Type23', u'Soil_Type24', u'Soil_Type25', u'Soil_Type26', u'Soil_Type27', u'Soil_Type28', u'Soil_Type29', u'Soil_Type30', u'Soil_Type31', u'Soil_Type32', u'Soil_Type33', u'Soil_Type34', u'Soil_Type35', u'Soil_Type36', u'Soil_Type37', u'Soil_Type38', u'Soil_Type39', u'Soil_Type40'
        ]

    xtrain = train_df.drop(labels=['Id','Cover_Type']+labels_to_drop, axis=1).values
    ytrain = transform_from_classes(train_df['Cover_Type'].values)
    #ytrain = train_df['Cover_Type'].values
    xtest = test_df.drop(labels=['Id']+labels_to_drop, axis=1).values
    ytest = ssub_df['Id'].values
    
    print xtrain.shape, ytrain.shape, xtest.shape, ytest.shape
    
    return xtrain, ytrain, xtest, ytest

def train_individual(xtrain, ytrain, xtest):
    #ytrain_pred = np.zeros(ytrain.shape[0])
    #ytest_pred = np.zeros((xtest.shape[0],ytrain.shape[1]))
    #model = MiniBatchKMeans(n_clusters=7)
    #ytest_pred = model.fit_predict(xtest)
    #ytrain_pred = model.predict(xtrain)
    #return np.hstack([xtrain, ytrain_pred.reshape(xtrain.shape[0],1)]), np.hstack([xtest, ytest_pred.reshape(xtest.shape[0],1)])
    
    pca = PCA(whiten=True)

    xtest = pca.fit_transform(xtest)
    xtrain = pca.transform(xtrain)
    

    return xtrain, xtest

def score_model(model, xtrain, ytrain):
    randint = reduce(lambda x,y: x|y, [ord(x)<<(n*8) for (n,x) in enumerate(os.urandom(4))])
    xTrain, xTest, yTrain, yTest = cross_validation.train_test_split(xtrain,
                                                                     ytrain,
                                                                     test_size=0.4, random_state=randint)
    model.fit(xTrain, yTrain)
    ytest_pred = model.predict(xTest)
    
    #print np.argmax(ytest_pred, axis=1)
    
    #pred_failed = ytest_pred != yTest
    #plot_failures(xTest[pred_failed], yTest[pred_failed])
    #print 'accuracy', accuracy_score(np.argmax(ytest_pred, axis=1), 
                                     #np.argmax(yTest, axis=1))
    
    #ytest_pred = np.zeros(yTest.shape)
    #for n in range(7):
        #model.fit(xTrain, yTrain[:,n])
        #ytest_pred[:,n] = model.predict(xTest)
        #print 'accuracy', accuracy_score(ytest_pred[:,n],yTest[:,n])
    #for x in range(ytest_pred.shape[0]):
        #nsel = ytest_pred[x,:].sum()
        #if nsel == 0:
            #npick = np.random.randint(6)
            #ytest_pred[x,npick] = 1
        #elif nsel > 1:
            #npick = np.random.randint(nsel)
            #tidx = 0
            #for n in range(6):
                #if ytest_pred[x,n] == 1:
                    #if tidx != npick:
                        #ytest_pred[x,n] = 0
                    #tidx += 1
    print 'accuracy', accuracy_score(ytest_pred,yTest)
    
    return model.score(xTest, yTest)

def prepare_submission(model, xtrain, ytrain, xtest, ytest):
    model.fit(xtrain, ytrain)
    ytest2 = transform_to_class(model.predict(xtest).astype(np.int64))
    #dateobj = map(datetime.datetime.fromtimestamp, ytest)
    
    df = pd.DataFrame({'Id': ytest, 'Cover_Type': ytest2}, columns=('Id', 'Cover_Type'))
    df.to_csv('submission.csv', index=False)
    
    return


if __name__ == '__main__':
    xtrain, ytrain, xtest, ytest = load_data()

    #model = RandomForestRegressor(n_jobs=-1)
    model = RandomForestClassifier(n_estimators=400, n_jobs=-1)
    #model = DecisionTreeClassifier()
    #model = GradientBoostingClassifier()
    #model = KNeighborsClassifier(7)
    
    #model = Pipeline([('kpca', KernelPCA(n_components=7)),
                      #('rf', RandomForestClassifier(n_estimators=400)),])
    
    #model = SVC()
    xtrain, xtest = train_individual(xtrain, ytrain, xtest)
    
    print 'score', score_model(model, xtrain, ytrain)
    #print model.feature_importances_
    prepare_submission(model, xtrain, ytrain, xtest, ytest)
