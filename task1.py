# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 22:44:31 2018

@author: Yogesh
"""
import numpy as np
import cv2
import random
from matplotlib import pyplot as plt
UBIT="yogeshja"
random.seed(sum([ord(c) for c in UBIT]))

image1 = cv2.imread('.\data\mountain1.jpg',0)         
image2 = cv2.imread('.\data\mountain2.jpg',0)

img1 = cv2.imread('.\data\mountain1.jpg')          
img2 = cv2.imread('.\data\mountain2.jpg') 

imge1= cv2.imread('.\data\mountain1.jpg')          
imge2 = cv2.imread('.\data\mountain2.jpg') 

#Reference:https://docs.opencv.org/3.1.0/da/df5/tutorial_py_sift_intro.html

gray1= cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray2= cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

sift=cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray1,None)
img3=cv2.drawKeypoints(gray1,kp,img1)
cv2.imwrite('task1_sift1.jpg',img3)
kp = sift.detect(gray2,None)
img4=cv2.drawKeypoints(gray2,kp,img2)
cv2.imwrite('task1_sift2.jpg',img4)

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(image1,None)
kp2, des2 = sift.detectAndCompute(image2,None)
#Reference:https://www.programcreek.com/python/example/89342/cv2.drawMatchesKnn

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)
# Apply ratio test
good = []
good_without_list = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append(m)
        good_without_list.append([m])

image3 = cv2.drawMatchesKnn(image1,kp1,image2,kp2,good_without_list,None, flags=2)
cv2.imwrite('task1_matches_knn.jpg',image3)
#Reference:https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
if len(good)>10:
    
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
 
count = 0
for i in range(len(matchesMask)):
    if(matchesMask[i]==1):
        if(count==10):
            index = i
            break
        count+=1

    h,w = image1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

else:
    print ("Not enough matches are found - %d",len(good))
    matchesMask = None
    
    
draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = random.sample(matchesMask,10), # draw only inliers
                   flags = 2)


image4=cv2.drawMatches(image1, kp1, image2, kp2, random.sample(good,10), None, **draw_params)
cv2.imwrite('task1_matches.jpg',image4)

#Reference:https://www.kaggle.com/asymptote/homography-estimate-stitching-two-imag

iH = np.linalg.inv(M)

def warpImages(image1, image2, M):
    
    ht1,wd1 = image1.shape[:2]
    p1 = np.float32([[0,0],[0,ht1],[wd1,ht1],[wd1,0]]).reshape(-1,1,2)
    ht2,wd2 = image2.shape[:2]
    p2 = np.float32([[0,0],[0,ht2],[wd2,ht2],[wd2,0]]).reshape(-1,1,2)
    
    p3 = cv2.perspectiveTransform(p2,M)
    
    p4 = np.concatenate((p1,p3), axis=0)
    
    [xmin, ymin] = np.int32(p4.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(p4.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    #Making use of translation
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) 

    result = cv2.warpPerspective(image1, Ht.dot(M), (xmax-xmin, ymax-ymin))
    result[t[1]:ht1+t[1],t[0]:wd1+t[0]] = image2
    return result

image5=warpImages(imge1, imge2, M)
cv2.imwrite('task1_pano.jpg',image5)