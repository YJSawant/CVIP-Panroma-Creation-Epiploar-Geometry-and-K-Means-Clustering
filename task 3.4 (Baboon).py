# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 13:06:54 2018

@author: Yogesh
"""

import numpy as np
import os
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('baboon.jpg')
pixels = np.asarray(img)
#width,height = img.shape 
reshapedImage=np.reshape(img,(-1,3))
pi=[]
pix=[]
for x in range(512):
    for y in range(512):
      r = pixels[x,y][0]
      g = pixels[x,x][1]
      b = pixels[x,x][2]
      pi=[r,b,g]
      pix.append(pi)
         
def compute_euclidean_distance(point, centroid):
    sum_of = 0
    for x in point:
        for y in centroid:
            ans = (x - y)**2
            sum_of += ans
    return (sum_of)**(1/2)

def assign_label_cluster(distance, data_point, centroids):
    index_of_minimum = min(distance, key=distance.get)
    return [index_of_minimum, data_point, centroids[index_of_minimum]]

def compute_new_centroids(c):
    m1 = 0
    m2 = 0
    m3 = 0
	#print(type(c[1][1]))
    for i in range(len(c)):
        for j in range(len(c[i])):
            if(j==0):
                m1 += c[i][j]
            elif(j==1):
                m2 += c[i][j]
            else:
                m3 += c[i][j]
    avg1 = m1/len(c)
    avg2 = m2/len(c)
    avg3 = m3/len(c)
    return np.round(avg1,0),np.round(avg2,0),np.round(avg3,0)

def iterate_k_means(data_points, centroids, total_iteration):
    label = []
    cluster_label = []
    total_points = len(data_points)
    k = len(centroids)
    cluster1=[]
    cluster2=[]
    cluster3=[]
    for iteration in range(0, total_iteration):
        for index_point in range(0, total_points):
            distance = {}
            for index_centroid in range(0, k):
                distance[index_centroid] = compute_euclidean_distance(data_points[index_point], centroids[index_centroid])
            label = assign_label_cluster(distance, data_points[index_point], centroids)
            if(label[0]==0):
                cluster1.append(label[1])
                
            elif(label[0]==1):
                cluster2.append(label[1])
            else:
                 cluster3.append(label[1])
            if iteration == (total_iteration-1):
                cluster_label.append(label)
    centroids[0] = compute_new_centroids(cluster1)
    centroids[1] = compute_new_centroids(cluster2)
    centroids[2] = compute_new_centroids(cluster3)
            #print (centroids)
    
            
    return [cluster_label, centroids]

def create_centroids():
    centroids = []
    centroids.append([78,80,68])
    centroids.append([115,130,109])
    centroids.append([56,83,74])
    return np.array(centroids)

if __name__ == "__main__":
    data_points =pix
    centroids = create_centroids()
    total_iteration = 10
    
    [cluster_label, new_centroids] = iterate_k_means(data_points, centroids, total_iteration)
    a = []
    cv = []
    for i in range(len(cluster_label)):
        a.append(cluster_label[i][1])
        cv.append(cluster_label[i][0])
    for i in range(len(cv)):
        a[i]= new_centroids[cv[i]]
        
    img1=np.reshape(a,(img.shape))
    cv2.imwrite('baboon_output_k3.jpg',img1)
    
    
    
    