import numpy as np
import cv2
import random
from matplotlib import pyplot as plt
UBIT="yogeshja"
random.seed(sum([ord(c) for c in UBIT]))

image1 = cv2.imread('tsucuba_left.png',0)          # queryImage
image2 = cv2.imread('tsucuba_right.png',0) # trainImage

img1 = cv2.imread('tsucuba_left.png')          # queryImage
img2 = cv2.imread('tsucuba_right.png') # trainImage

imge1= cv2.imread('tsucuba_left.png',0)          # queryImage
imge2 = cv2.imread('tsucuba_right.png',0) # trainImage

im1= cv2.imread('tsucuba_left.png',0)          # queryImage
im2 = cv2.imread('tsucuba_right.png',0) # trainImage

#Reference:https://docs.opencv.org/3.1.0/da/df5/tutorial_py_sift_intro.html

gray1= cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray2= cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

# Initiate SIFT detector
#sift = cv2.SIFT()
sift=cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray1,None)
img3=cv2.drawKeypoints(gray1,kp,img1)
cv2.imwrite('task2_sift1.jpg',img3)
kp = sift.detect(gray2,None)
img4=cv2.drawKeypoints(gray2,kp,img2)
cv2.imwrite('task2_sift2.jpg',img4)

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
pts1=[]
pts2=[]

for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append(m)
        good_without_list.append([m])
        
        pts1.append(kp1[m.queryIdx].pt)
        pts2.append(kp2[m.trainIdx].pt)
        
# cv.drawMatchesKnn expects list of lists as matches.
image3 = cv2.drawMatchesKnn(image1,kp1,image2,kp2,good_without_list,None, flags=2)
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
cv2.imwrite('task2_matches_knn.jpg',image3)

#Reference:https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_epipolar_geometry/py_epipolar_geometry.html
F, mask2 = cv2.findFundamentalMat(pts1,pts2,cv2.RANSAC)
#plt.imshow(img3)
#plt.show()

# We select only inlier points
pts1 = pts1[mask2.ravel()==1]
pts2 = pts2[mask2.ravel()==1]

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2
# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
    
epipts1=np.asarray(random.sample(list(pts1),10))
epipts2=np.asarray(random.sample(list(pts2),10))

lines1 = cv2.computeCorrespondEpilines(epipts2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
epilines1 = np.asarray(random.sample(list(lines1),10))

im5,im6 = drawlines(im1,im2,lines1,epipts1,epipts2)
# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(epipts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
epilines2 = np.asarray(random.sample(list(lines2),10))

im3,im4 = drawlines(im2,im1,lines2,epipts2,epipts1)

plt.subplot(121),plt.imshow(im5)
plt.subplot(122),plt.imshow(im3)
plt.show()
cv2.imwrite('task2_epi_right.jpg',im3)
cv2.imwrite('task2_epi_left.jpg',im5)

#Reference1=https://docs.opencv.org/3.4/d2/d85/classcv_1_1StereoSGBM.html
#Reference2=https://docs.opencv.org/3.1.0/dd/d53/tutorial_py_depthmap.html

window_size = 5
min_disp = 0
num_disp = 64-min_disp
stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = 11,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        disp12MaxDiff = 2,
        preFilterCap=2,
        uniquenessRatio = 5,
        speckleWindowSize = 200,
        speckleRange = 1
    )

disp = stereo.compute(imge1, imge2).astype(np.float32) / 16.0
#cv2.imshow('task2 disparity.jpg',disp)

plt.imshow(disp,'gray')
plt.imsave("task2_disparity.jpg",disp,cmap='gray')
plt.show()
