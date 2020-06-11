# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 12:30:31 2020

@author: Varun
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from	sklearn.cluster	import	KMeans
from scipy.spatial.distance import cdist
lines= pd.read_excel("EastWestAirlines.xlsx",'data')
lines
lines.shape
###kmeans method
def norm_func(i):
    x = (i-i.min())	/	(i.max()	-	i.min())
    return (x)
df_norm=norm_func(lines.iloc[:,1:])
df_norm.head(10)
k=list(range(2,15))
k
TWSS=[]
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(df_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))
    
    
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)
###from graph, k=5
model=KMeans(n_clusters=5)
model.fit(df_norm)
model.labels_
md=pd.Series(model.labels_)
lines['clust']=md
lines
liness=lines.iloc[:,[12,0,1,2,3,4,5,6,7,8,9,10,11]]
liness.iloc[:,1:12].groupby(lines.clust).mean()
liness.to_excel("EastWestAirliness.xlsx",'data')



##hierarchiual method
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
air=pd.read_excel("EastWestAirlines.xlsx",'data')
air
def norm_func(i):
    x = (i-i.min())	/	(i.max()	-	i.min())
    return (x)

df_norm=norm_func(air.iloc[:,1:])
df_norm.head()


from scipy.cluster.hierarchy import linkage
###for creating dendograms
import scipy.cluster.hierarchy as sch

type(df_norm)
help(linkage)
z=linkage(df_norm,method="complete", metric="euclidean")
plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()






from	sklearn.cluster	import	AgglomerativeClustering 
h_complete	=	AgglomerativeClustering(n_clusters=3,	linkage='complete',affinity = "euclidean").fit(df_norm) 

h_complete.labels_

cluster_labels=pd.Series(h_complete.labels_)
cluster_labels.value_counts()

air['clust']=cluster_labels # creating a  new column and assigning it to new column 
air = air.iloc[:,[12,0,1,2,3,4,5,6,7,8,9,10,11]]
air.head()

# getting aggregate mean of each cluster
air.groupby(air.clust).mean()

# creating a csv file 
air.to_csv("airlineshierarchial.csv") #,encoding="utf-8")
