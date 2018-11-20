# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 20:00:44 2018
@author: Yogesh
"""
import numpy as np
import cv2
import random
import math
from matplotlib import pyplot as plt

def eucldist(point,centroid):
    dist = 0.0
    for i in range(0,len(point)):
        dist += (point[i] - centroid[i])**2
    return math.sqrt(dist)

def create_centroids():
    centroids = []
    centroids.append([6.2, 3.2])
    centroids.append([6.6, 3.7])
    centroids.append([6.5, 3.0])
    return np.array(centroids)

def data_points():
    data_points=[]
    data_points.append([5.9, 3.2])
    data_points.append([4.6, 2.9])
    data_points.append([6.2, 2.8])
    data_points.append([4.7, 3.2])
    data_points.append([5.5, 4.2])
    data_points.append([5.0, 3.0])
    data_points.append([4.9, 3.1])
    data_points.append([6.7, 3.1])
    data_points.append([5.1, 3.8])
    data_points.append([6.0, 3.0])
    return np.array(data_points)

centroids = create_centroids()
data_points =data_points()

a, b= centroids.T
plt.scatter(a[0],b[0],marker=".",c="red")
plt.scatter(a[1],b[1],marker=".",c="green")
plt.scatter(a[2],b[2],marker=".",c="blue")



def turn_to_dict(*args):
    return {i: v for i, v in enumerate(args)}

cluster1=[]
cluster2=[]
cluster3=[]

def new_centroid(cluster):
    mx = 0
    my = 0
    for i in range(len(cluster)):
        for j in range(len(cluster[i])):
            if(j==0):
                mx += cluster[i][j]
            else:
                my += cluster[i][j]
    avg1 = mx/len(cluster)
    avg2 = my/len(cluster)
    return np.round(avg1,1),np.round(avg2,1)

vector=[]
def assign_min_cluster(dist,datapoint,centroids):
    index_of_minimum = min(dist, key=dist.get)

    if(index_of_minimum==0):
        #plt.scatter(datapoint[0], datapoint[1], c='red', s=50 , marker='^')
        cluster1.append(datapoint)
        vector.append(0)
    elif(index_of_minimum==1):
        #plt.scatter(datapoint[0], datapoint[1], c='green', s=50 , marker='^')
        cluster2.append(datapoint)
        vector.append(1)
    else:
        #plt.scatter(datapoint[0], datapoint[1], c='blue', s=50 , marker='^')
        cluster3.append(datapoint)
        vector.append(2)
        
    cluster = turn_to_dict(cluster1, cluster2, cluster3)
    return index_of_minimum,cluster,cluster1,cluster2,cluster3,vector

def k_means(data_points, centroids,iteration):
    cluster=[]
    cluster1=[]
    cluster2=[]
    cluster3=[]
    vector=[]
    for i in range(0, iteration):
        for j in range(0, len(data_points)):
            dist = {}
            for k in range(0, 3):
                dist[k] = eucldist(data_points[j], centroids[k])
            index_of_minimum,cluster,cluster1,cluster2,cluster3,vector=assign_min_cluster(dist, data_points[j], centroids)
    
    return cluster,cluster1,cluster2,cluster3,vector

clusterone=[]
#cluster=k_means(data_points, centroids,1)
clusterone1=[]
clusterone2=[]
clusterone3=[]
vector1=[]
clusterone,clusterone1,clusterone2,clusterone3,vector=k_means(data_points, centroids,1)
print (vector)

clusterone1=np.array(clusterone1)
a1,b1=clusterone1.T
plt.scatter(a1, b1, c='red', s=50 , marker='^')
clusterone2=np.array(clusterone2)
a1,b1=clusterone2.T
plt.scatter(a1, b1, c='green', s=50 , marker='^')
clusterone3=np.array(clusterone3)
a1,b1=clusterone3.T
plt.scatter(a1, b1, c='blue', s=50 , marker='^')

plt.savefig('task3_iter1_a.jpg')
plt.figure()

centroids2=[]
centroids2.append(list(new_centroid(clusterone1)))
centroids2.append(list(new_centroid(clusterone2)))
centroids2.append(list(new_centroid(clusterone3))) 
centroids2=np.array(centroids2)

a, b= centroids2.T
plt.scatter(a[0],b[0],marker=".",c="red")
plt.scatter(a[1],b[1],marker="o",s=40,c="green")
plt.scatter(a[2],b[2],marker=".",c="blue")

clusterone1=np.array(clusterone1)
a1,b1=clusterone1.T
plt.scatter(a1, b1, c='red', s=50 , marker='^')
clusterone2=np.array(clusterone2)
a1,b1=clusterone2.T
plt.scatter(a1, b1, c='green', s=50 , marker='^')
clusterone3=np.array(clusterone3)
a1,b1=clusterone3.T
plt.scatter(a1, b1, c='blue', s=50 , marker='^')

plt.savefig('task3_iter1_b.jpg')
plt.figure()

clustertwo=[]
#cluster=k_means(data_points, centroids,1)
clustertwo1=[]
clustertwo2=[]
clustertwo3=[]
cluster1=[]
cluster2=[]
cluster3=[]
vector=[]
vector2=[]

clustertwo,clustertwo1,clustertwo2,clustertwo3,vector2=k_means(data_points, centroids2,1)
print (vector2)

a, b= centroids2.T
plt.scatter(a[0],b[0],marker=".",c="red")
plt.scatter(a[1],b[1],marker="o",s=40,c="green")
plt.scatter(a[2],b[2],marker=".",c="blue")

clustertwo1=np.array(clustertwo1)
a1,b1=clustertwo1.T
plt.scatter(a1, b1, c='red', s=50 , marker='^')
clustertwo2=np.array(clustertwo2)
a1,b1=clustertwo2.T
plt.scatter(a1, b1, c='green', s=50 , marker='^')
clustertwo3=np.array(clustertwo3)
a1,b1=clustertwo3.T
plt.scatter(a1, b1, c='blue', s=50 , marker='^')

plt.savefig('task3_iter2_a.jpg')
plt.figure()

centroids3=[]
centroids3.append(list(new_centroid(clustertwo1)))
centroids3.append(list(new_centroid(clustertwo2)))
centroids3.append(list(new_centroid(clustertwo3))) 
centroids3=np.array(centroids3)

a, b= centroids3.T
plt.scatter(a[0],b[0],marker=".",c="red")
plt.scatter(a[1],b[1],marker="o",s=40,c="green")
plt.scatter(a[2],b[2],marker=".",c="blue")

clustertwo1=np.array(clustertwo1)
a1,b1=clustertwo1.T
plt.scatter(a1, b1, c='red', s=50 , marker='^')
clustertwo2=np.array(clustertwo2)
a1,b1=clustertwo2.T
plt.scatter(a1, b1, c='green', s=50 , marker='^')
clustertwo3=np.array(clustertwo3)
a1,b1=clustertwo3.T
plt.scatter(a1, b1, c='blue', s=50 , marker='^')

plt.savefig('task3_iter2_b.jpg')

