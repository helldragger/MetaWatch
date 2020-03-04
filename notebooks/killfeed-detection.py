#%%
# 
# 
import cv2

import numpy as np;

from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd


def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


img = cv2.imread("data/raw/3/truth.PNG", cv2.IMREAD_UNCHANGED)

#%%

import numpy as np
import cv2
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10

img1 = cv2.imread('data/raw/3/hamster_icone.PNG',0)          # queryImage
img2 = cv2.imread('data/raw/3/truth.PNG',0) # trainImage

# Initiate SIFT detector
orb = cv2.ORB_create(nfeatures=1500)

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

img1KpPreview = cv2.drawKeypoints(img1, kp1, None)
cv2.imshow("Image 1 keypoints", img1KpPreview)
img2KpPreview = cv2.drawKeypoints(img2, kp2, None)
cv2.imshow("Image 2 keypoints ", img2KpPreview)
cv2.waitKey(0);
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

else:
    print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

plt.imshow(img3, 'gray'),plt.show()


#%%


#img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#cv2.imshow("img layer 0 before contrasting", img[:,:,0]);
#apply_brightness_contrast(img[:,:,0], contrast=50, brightness=-50);
#cv2.imshow("img layer 0 after contrasting", img[:,:,0]);
#contours, hierarchy = cv2.findContours(img[:,:,0], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE );
#cv2.drawContours(img,contours,-1, [0, 255, 0]);
#img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
#import pytesseract
#plt.imshow(img[:,:,0])
#plt.show();
#text = pytesseract.image_to_string(img, config="-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 6")
#from pytesseract import Output
#data = pytesseract.image_to_data(img[:,:,0], config="-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ", output_type=Output.DATAFRAME)

#cv2.imshow("contoured img", img);
#cv2.waitKey(0);

#%%
