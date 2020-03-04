#%%
import numpy as np
import cv2
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join
import seaborn as sns
import pytesseract

imgDir = "src/data/tests/Ultimate/"
onlyfiles = [f for f in listdir(imgDir) if isfile(join(imgDir, f))]


#imgHeatmap = None;
#lastImg = None;
#for fileSrc in onlyfiles:
 #   img = cv2.imread(imgDir+"/"+fileSrc)
 #   img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
 #   if imgHeatmap is None:
 #       imgHeatmap=np.zeros(img.shape)
 #       lastImg = img
 #   
#    if img.shape != imgHeatmap.shape:
#        continue
#    else:
#        imgHeatmap = imgHeatmap + img

#sns.heatmap(imgHeatmap);
#plt.show();
#thresholedHeatmap = imgHeatmap
#thresholedHeatmap[thresholedHeatmap> 110000] = 0
#sns.heatmap(thresholedHeatmap);

#plt.show();

#plt.show();
#cv2.imshow("heatmap",imgHeatmap)
#cv2.waitKey(0)

#%%

#%%




import math
img1 = cv2.imread('src/data/tests/Ultimate/100 (1).png',0)          # queryImage
img2 = cv2.imread('src/data/tests/Ultimate/100 (2).png',0) # trainImage


#%%
# Initiate feature detector
kaze = cv2.KAZE_create(extended=True, upright=True , threshold=1e-6)

# find the keypoints and descriptors with SIFT
kp1, des1 = kaze.detectAndCompute(img1,None)
kp2, des2 = kaze.detectAndCompute(img2,None)
#img1KpPreview = cv2.drawKeypoints(img1, kp1, None)
#cv2.imshow("Image 1 keypoints", img1KpPreview)
#img2KpPreview = cv2.drawKeypoints(img2, kp2, None)
#cv2.imshow("Image 2 keypoints ", img2KpPreview)

#print("Visualizing keypoints...")
#cv2.waitKey(0);
#%%

search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(None, search_params)

matches = flann.knnMatch(des1,des2,k=2)
MIN_MATCH_COUNT = math.floor(len(kp1)*.7)
# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)
if len(good)>=MIN_MATCH_COUNT: # aiming for 70% probability of detection
    print("Ultimate detected!")
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   #matchesMask = matchesMask, # draw only inliers
                   flags = 2)
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    plt.imshow(img3, 'gray'),plt.show()
else:
    print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
    matchesMask = None



#%%


def convertUltimateImgToText(img):
    plt.imshow(img);
    plt.show();
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY )
    
    plt.imshow(img);
    plt.show();
    
    srcTri = np.array( [[0, 0], [img.shape[1] - 1, 0], [0, img.shape[0] - 1]] ).astype(np.float32)
    dstTri = np.array( [[0, 0], [img.shape[1] - 1, 0], [img.shape[1]*0.19, img.shape[0] - 1]] ).astype(np.float32)
    warp_mat = cv2.getAffineTransform(srcTri, dstTri)
    img = cv2.warpAffine(img, warp_mat, (img.shape[1], img.shape[0]))
    plt.imshow(img);
    plt.show();
    
    thresh = 200
    maxValue = 255
    img = cv2.threshold(img, thresh, maxValue, cv2.THRESH_BINARY )[1]
    plt.imshow(img);
    plt.show();
    text = pytesseract.image_to_string(img, config="tessedit_write_images=true  digits", lang="Koverwatch")

    print("img ult text: '{}'".format(text))
    return text


def testUltimateParsing():

    imgDir = "src/data/tests/Ultimate/"
    onlyfiles = [f for f in listdir(imgDir) if isfile(join(imgDir, f)) and f.split(" ")[0]=="100"]
    
    TRUTH_imgs = []
    # Initiate feature detector
    kaze = cv2.KAZE_create(extended=True, upright=True , threshold=1e-6)
    

    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(None, search_params)

    # test format: COLOR_expectedValue.png
    
    expected = {}
    testedValues = set();
    valueToIndex = {}
    valueLabels = []
    for src in onlyfiles:
        value  = src.split(" ")[0];
        expected[src] = (value == "100");
        if expected[src]: # those images will be used as truth values
            TRUTH_img = cv2.imread(join(imgDir, src),-1);
            TRUTH_kp, TRUTH_des = kaze.detectAndCompute(TRUTH_img,None)
            TRUTH_imgs.append((src, TRUTH_kp, TRUTH_des))

        if value not in testedValues:
            valueToIndex[src] = len(testedValues)
            testedValues.add(src)
            valueLabels.append(src)

    
    errors = np.zeros((len(testedValues),  2))
    # we consider 80% of the reference images must agree to consider a detection succesful.
    MIN_DETECTION_COUNT = math.floor(len(TRUTH_imgs) * .8)

    for src in expected.keys():
        TEST_img = cv2.imread('src/data/tests/Ultimate/'+src,-1);
        TEST_kp, TEST_des = kaze.detectAndCompute(TEST_img, None)
        value = 0
        for TRUTH_src, TRUTH_kp, TRUTH_des in TRUTH_imgs:
            matches = flann.knnMatch(TRUTH_des,TEST_des,k=2)
            MIN_MATCH_COUNT = math.floor(len(TRUTH_kp)*.8) # 70% des features de l'image de reference doivent coller. 
            # store all the good matches as per Lowe's ratio test.
            good = []
            for m,n in matches:
                if m.distance < 0.7*n.distance:
                    good.append(m)
            # here we add 1 to value if the current reference img detects a ultimate
            value+= 1 if len(good)>=MIN_MATCH_COUNT else 0
        # here we consider the detection valid if enough reference images detected it
        value = value>=MIN_DETECTION_COUNT
        if value != expected[src]:
            errors[valueToIndex[src],  0 if value else 1] += 1
    

    sns.heatmap(data=errors, xticklabels=["Expected: False", "True"], yticklabels=valueLabels)#.get_figure()#.savefig("src/data/tests/Ultimate/Tests_ErrorHeatmap.png")
    plt.show()
    totalErrors = errors.sum(axis=0).sum()
    
    ultimateAcc = 1 - (totalErrors / len(onlyfiles))
    print("total errors:", totalErrors ,"/", errors.shape[0])
    print("Ultimate detection accuracy: ", ultimateAcc)
    assert ultimateAcc >= 0.90 # 10% d'erreur
    #return errors, valueLabels, valueToIndex



testUltimateParsing()

#%%


indexToValue = {valueToIndex[key]:key for key in valueToIndex.keys()}
nonzeroIndexes = np.where(errors != [0,0])
#%%

list(map( lambda index, indexToValue=indexToValue: indexToValue[index] , nonzeroIndexes[0]))




#%%




imgToLinify = cv2.imread('C:/Users/Helldragger/Pictures/ForFurRiod.png',cv2.IMREAD_GRAYSCALE)

cv2.createLineSegmentDetector().detect(imgToLinify)
