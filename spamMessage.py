@@ -0,0 +1,72 @@
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 10:09:30 2018

@author: moulf
"""

import pandas as pd
import numpy as np
import jieba
import itertools 
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def train_predict(clf,X_train_tfidf,train_y,X_test_tfidf,test_y):
    clf.fit(X_train_tfidf,train_y)
    predicted_categories = clf.predict(X_test_tfidf)
    score = accuracy_score(test_y,predicted_categories)
    print(score)
    confusion(test_y,predicted_categories)
    

def confusion(test_y,predicted_categories):
    NB_confusion = confusion_matrix(test_y ,predicted_categories)
    plt.imshow(NB_confusion,interpolation='nearest',cmap=plt.cm.Oranges)
    plt.xlabel('predicted')
    plt.ylabel('True')
    tick_marks = np.arange(2)
    plt.xticks(tick_marks,[0,1],rotation= 45)
    plt.yticks(tick_marks,[0,1])
    plt.colorbar()
    plt.title('confustion_matrix')
    for i,j in itertools.product(range(len(NB_confusion)),range(len(NB_confusion))):
        plt.text(i,j,NB_confusion[j,i],
                 horizontalalignment="center")
    plt.show()
def vector_space():
    data = pd.read_csv(r'data\80w.txt',encoding = 'utf-8', sep='	',header=None)
    stopwords = pd.read_csv(r'data\hlt_stop_words.txt',  encoding='utf-8',sep='\r\n')
    data['分词短信'] = data[2].apply(lambda x:' '.join(jieba.cut(x)))
    X = data['分词短信'].values
    y = data[1].values
    train_X,test_X,train_y,test_y = train_test_split(X,y,test_size=0.1,random_state=100)
    
    #训练
    vectorizer = CountVectorizer(stop_words=list(stopwords))
    X_train_termcounts = vectorizer.fit_transform(train_X)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_termcounts)
    X_test_termcounts = vectorizer.transform(test_X)
    X_test_tfidf = tfidf_transformer.transform(X_test_termcounts)
    clf = MultinomialNB(alpha=0.001)
    #预测
    print('MultinomialNB:')
    train_predict(clf,X_train_tfidf,train_y,X_test_tfidf,test_y)
    #clf = KNeighborsClassifier(n_neighbors=5,algorithm='brute')
    print('KNN:')
    #train_predict(clf,X_train_tfidf,train_y,X_test_tfidf,test_y)
    clf = SVC()
    print('SVC:')
    train_predict(clf,X_train_tfidf,train_y,X_test_tfidf,test_y)


if __name__ == '__main__': 
    vector_space()