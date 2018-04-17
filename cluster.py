@@ -0,0 +1,49 @@
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 17:09:47 2018

@author: moulf
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans,DBSCAN,AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

#绘制聚类结果
def showClusterPlt(tsne,label_pred,title):
    plt.title(title)
    plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
    d = tsne[label_pred == 0]
    plt.scatter(d[0], d[1], c = "red", marker='o', label='label0') 
    d = tsne[label_pred == 1]
    plt.scatter(d[0], d[1], c = "green", marker='*', label='label1')
    d = tsne[label_pred == 2]
    plt.scatter(d[0], d[1], c = "blue", marker='+', label='label2')
    plt.legend(loc=0)
    plt.show()
def runCluster():    
    data = pd.read_csv(r'data\HR_comma_sep.csv',encoding='utf-8',header= 0)
    data = data.dropna()
    
    X = data.iloc[:,0:8]
    #进行数据降维
    tsne = TSNE()
    tsne.fit_transform(X) 
    tsne = pd.DataFrame(tsne.embedding_, index = X.index) #转换数据格式
    #k-means聚类
    title = 'k-means聚类'
    cluseter= KMeans(n_clusters=3,random_state=170)
    cluseter.fit_predict(X)
    #不同类别用不同颜色和样式绘图
    label_pred = cluseter.labels_ #获取聚类标签
    showClusterPlt(tsne,label_pred,title)
    
    title = '层次聚类'
    cluseter= AgglomerativeClustering(affinity='euclidean',linkage='ward',n_clusters=3)
    cluseter.fit_predict(X)
    label_pred = cluseter.labels_ #获取聚类标签
    showClusterPlt(tsne,label_pred,title)
if __name__ == '__main__':
    runCluster()