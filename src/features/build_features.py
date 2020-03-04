#%%


import cv2;

from pathlib import Path
from dotenv import find_dotenv, load_dotenv

# not used in this stub but often useful for finding various files
#project_dir = Path(__file__).resolve().parents[2]

# find .env automagically by walking up directories until it's found, then
# load up the .env entries as environment variables
load_dotenv(find_dotenv())
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd 
from scipy import ndimage
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

import seaborn as sns
img = cv2.imread("C:\\Users\\Helldragger\\Documents\\projects\\MetaWatch\\MetaWatch\\src\\features\\original.jpg");
# we cut the image to keep only the interesting part: the overlay.

#%%


bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

newmask = cv.imread('C:\\Users\\Helldragger\\Documents\\projects\\MetaWatch\\MetaWatch\\src\\features\\overlaymaskheroes.png',0)
# wherever it is marked white (sure foreground), change mask=1
# wherever it is marked black (sure background), change mask=0
mask[newmask == 0] = 0
mask[newmask == 255] = 1
mask, bgdModel, fgdModel = cv.grabCut(img,mask,None,bgdModel,fgdModel,5,cv.GC_INIT_WITH_MASK)
mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img2 = img*mask[:,:,np.newaxis]
#%%
plt.imshow(img),plt.colorbar(),plt.show()
plt.imshow(img2),plt.colorbar(),plt.show()

cv2.imshow("cut", img2);
cv2.imshow("original", img);
cv2.waitKey(0)


croppedImg = img2[125:210, 35:1892];
plt.imshow(croppedImg),plt.colorbar(),plt.show()

cv2.imshow("cropped", croppedImg);
cv2.waitKey(0)

#%%
# read data about a single hero
src1_mask = cv2.imread('C:\\Users\\Helldragger\\Documents\\projects\\MetaWatch\\MetaWatch\\src\\features\\maskheroA1.png',0);
src1_mask = cv2.cvtColor(src1_mask,cv2.COLOR_GRAY2RGB)    
masked_image = cv2.bitwise_and(img, src1_mask)
cv2.imshow("hero maked", masked_image);
cv2.waitKey(0)

#%%
#cropping the hhero image

y,x = masked_image[:,:,1].nonzero() # get the nonzero alpha coordinates
minx = np.min(x)
miny = np.min(y)
maxx = np.max(x)
maxy = np.max(y) 

cropImg = masked_image[miny:maxy, minx:maxx]

#cv2.imwrite("cropped.png", cropImg)
cv2.imshow("cropped", cropImg)
cv2.waitKey(0)


#%%

# here we load various  health bars and try to quantify their content: health, shield, armor, death symbol.

imgs = {};
srcs = [
    "frame377_hero_A1_health",
    "frame377_hero_A2_health",
    "frame377_hero_A3_health",
    "frame377_hero_A4_health",
    "frame377_hero_A5_health",
    "frame377_hero_A6_health",
    "frame377_hero_B1_health",
    "frame377_hero_B2_health", 
    "frame377_hero_B3_health", 
    "frame377_hero_B4_health", 
    "frame377_hero_B5_health", 
    "frame377_hero_B6_health",
    ]
for src in srcs:
    imgs[src] = cv2.imread('C:\\Users\\Helldragger\\Documents\\projects\\MetaWatch\\MetaWatch\\src\\data\\tests\\'+src+'.jpg',-1);
#src1_mask = cv2.imread('C:\\Users\\Helldragger\\Documents\\projects\\MetaWatch\\MetaWatch\\src\\features\\maskheroA1.png',0);

#%%

# here we create an histogram for each image
# idea: counting the amount of health using the percentage of specific colours in the health zone.
img = imgs["frame377_hero_B1_health"];

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB )
img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS )

channels = ('H','L','S')
colors = ("B", 'G', 'R')
for i,chann in enumerate(channels):
    histr = cv2.calcHist(img, [i], None, [256], [1, 256], True, False)
    histr /= max(histr)
    plt.plot(histr,color = colors[i])
    plt.xlim([0,256])

plt.show()
plt.imshow(img),plt.colorbar(),plt.show()
#for img in imgs.values():

#%%
# Reducing the amount of colors to 5 colors:
reducedimg = imgs["frame377_hero_B1_health"];
#cv2.cvtColor(reducedimg, cv2.COLOR_HLS2BGR )
plt.imshow(reducedimg),plt.colorbar(),plt.show()
reducedimg = reducedimg // 64
reducedimg = reducedimg * 64
plt.imshow(reducedimg),plt.colorbar(),plt.show()
#%%
img = imgs["frame377_hero_B1_health"];

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB )

pixel_colors = img.reshape((np.shape(img)[0]*np.shape(img)[1], 3))
norm = colors.Normalize(vmin=-1.,vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()

hlsImg = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
h, l, s = cv2.split(hlsImg);



#plt.imshow(img),plt.colorbar()
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1, projection="3d")
ax1.scatter(h.flatten(), l.flatten(), s.flatten(), facecolors=pixel_colors, marker=".")
ax1.set_xlabel("Hue")
ax1.set_ylabel("Luminosity")
ax1.set_zlabel("Saturation")

ax2 = fig.add_subplot(2, 1, 2)
ax2.imshow(img)
plt.show()

#%%
# trying to filter out noise colors:
img = imgs["frame377_hero_B1_health"];

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB )

hlsImg = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
h, l, s = cv2.split(hlsImg);

fig = plt.figure()

ax = fig.add_subplot(4, 1, 1)
ax.imshow(img)
ax.set_label("pre filtering");

h, l, s = cv2.split(hlsImg);


ax = fig.add_subplot(4, 1, 2)
ax.imshow(h)
ax.set_label("Hue");

ax = fig.add_subplot(4, 1, 3)
ax.imshow(l)
ax.set_label("Luminance");

ax = fig.add_subplot(4, 1, 4)
ax.imshow(s)
ax.set_label("Saturation");
plt.show()

#%%

def getRGBImg(name):
    return cv2.cvtColor(imgs[name], cv2.COLOR_BGR2RGB)

def showHealthDetected(img):
    hlsImg = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    hlsImg[hlsImg[:,:,1] < 100] = 0
    #hlsImg[:,:,0] = 126
    h, l, s = cv2.split(hlsImg);


    fig = plt.figure()

    ax = fig.add_subplot(5, 1, 1)
    ax.imshow(img)

    ax = fig.add_subplot(5, 1, 2)
    ax.imshow(h)
    ax.set_label("Hue");

    ax = fig.add_subplot(5, 1, 3)
    ax.imshow(l)
    ax.set_label("Luminance");

    ax = fig.add_subplot(5, 1, 4)
    ax.imshow(s)
    ax.set_label("Saturation");

    resultimg = cv2.cvtColor(hlsImg, cv2.COLOR_HLS2RGB)

    ax = fig.add_subplot(5, 1, 5)
    ax.imshow(resultimg)
    plt.show()

    return resultimg

analyzedImgs = {}
for imgSrc in imgs.keys():
    analyzedImgs[imgSrc] = showHealthDetected(getRGBImg(imgSrc))

#%%
# recenter every img.
# calculate the histogram of pixels > 0  per line.
def showLineHistograms(img):
    img[img > 0] = 1
    Ysize = len(img[:,0,0])
    hist = img.sum(axis=1)
    fig = plt.figure()
    plt.plot(hist)
    plt.show()




hist = np.zeros((22));
observations = np.array([]);
for imgSrc in imgs.keys():
    img = cv.cvtColor(analyzedImgs[imgSrc],  cv2.COLOR_RGB2GRAY)
    imgSum = img.sum(axis=1);
    hist[: len(imgSum)] += imgSum
    observations = np.concatenate((observations, np.where(imgSum > 0)[0]) );


hist /= max(hist)
fig = plt.figure()
plt.plot(hist)
plt.show()

sns.distplot(observations)   
plt.show()

#%%
# here we try to detect the correct amount of notches on any image
# the image is passed through a gradient filter to show only the variations.
# this gradient filter is then 
def detectLabels(img):
    return ndimage.measurements.label(imgGrad)

def calculateLabellingErrorRate(gradImg, expected):
    return detectLabels(gradImg)[1] - expected;

notchesExpected = {
    "frame377_hero_A1_health":23,
    "frame377_hero_A2_health":24,
    "frame377_hero_A3_health":8,
    "frame377_hero_A4_health":10,
    "frame377_hero_A5_health":2,
    "frame377_hero_A6_health":6,
    "frame377_hero_B1_health":24,
    "frame377_hero_B2_health":8, 
    "frame377_hero_B3_health":10, 
    "frame377_hero_B4_health":8, 
    "frame377_hero_B5_health":1, 
    "frame377_hero_B6_health":8,
}
gradientThreshold = 30
bin_amount = abs(min(256, 256))
gradientThresholds = np.linspace(0,256,bin_amount) // 1


#%%
errors = np.zeros(shape=(len(notchesExpected), bin_amount))
j = 0
for lowGradientThreshold in gradientThresholds:
    i = 0
    for src in imgs.keys():
        expected = notchesExpected[src];
        img = imgs[src];
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB )
        imgGrad = ndimage.morphology.morphological_gradient(img[:,:,0], size=(1,2))
        imgGrad[imgGrad<lowGradientThreshold] = 0
        labels, count = ndimage.measurements.label(imgGrad)
        errors[i, j] = abs(calculateLabellingErrorRate(img, expected))
        i += 1
    j+=1

sns.heatmap(errors)
plt.plot()
plt.figure()
errorsSummed = errors.sum(axis=0);
errorsSummed
sns.lineplot(data=errorsSummed)
plt.plot()
#plot.figure()
bestThreshold = np.where(errorsSummed == min(errorsSummed))[0][-1];
print("best high pass gradient threshold: ",bestThreshold, "\n\terror count:", errorsSummed[bestThreshold])
# low pass gradient gave a best score of 60 errors, not enough. combining both systems gave a best score of 38. we will keep the simple high pass.
#%%


# best system, simple high filter, with a threshold of 52 or around 50




#%%
img = imgs["frame377_hero_B1_health"];

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB )

plt.imshow(img),plt.colorbar(),plt.show()
imgGrad = ndimage.morphology.morphological_gradient(img[:,:,0], size=(1,2))
plt.imshow(imgGrad),plt.colorbar(),plt.show()

imgGrad[imgGrad<30] = 0
plt.imshow(imgGrad),plt.colorbar(),plt.show()

label, num_features = ndimage.measurements.label(imgGrad)
print("###RGB mode on gradient")
print(num_features, " features detected / 24.")


HLSImg = cv.cvtColor(analyzedImgs[imgSrc],  cv.COLOR_RGB2HLS)


label, num_features = ndimage.measurements.label(HLSImg[:,:,0])
print("###HLSImg using Hue")
print(num_features, " features detected / 24.")
label, num_features = ndimage.measurements.label(HLSImg[:,:,1])
print("###HLSImg using Luminosity")
print(num_features, " features detected / 24.")
label, num_features = ndimage.measurements.label(HLSImg[:,:,2])
print("###HLSImg using Saturation")
print(num_features, " features detected / 24.")
#%%
# here we try to differentiate notches types.

notchesTypesExpected = {
    "frame377_hero_A1_health":{"white":20,"yellow":3, "blue":0, "red":0},
    "frame377_hero_A2_health":{"white":17,"yellow":7, "blue":0, "red":0},
    "frame377_hero_A3_health":{"white":8,"yellow":0, "blue":0, "red":0},
    "frame377_hero_A4_health":{"white":8,"yellow":2, "blue":0, "red":0},
    "frame377_hero_A5_health":{"white":2,"yellow":0, "blue":0, "red":0},
    "frame377_hero_A6_health":{"white":2,"yellow":0, "blue":4, "red":0},
    "frame377_hero_B1_health":{"white":20,"yellow":4, "blue":0, "red":0},
    "frame377_hero_B2_health":{"white":8,"yellow":0, "blue":0, "red":0}, 
    "frame377_hero_B3_health":{"white":8,"yellow":2, "blue":0, "red":0}, 
    "frame377_hero_B4_health":{"white":8,"yellow":0, "blue":0, "red":0}, 
    "frame377_hero_B5_health":{"white":0,"yellow":0, "blue":0, "red":1}, 
    "frame377_hero_B6_health":{"white":8,"yellow":0, "blue":0, "red":0},
}
#%%



img = imgs["frame377_hero_A6_health"];
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB )
originalimg = img

#plt.imshow(img),plt.colorbar(),plt.show()

imgHLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS )
#plt.imshow(imgHLS[:,:,1]),plt.colorbar(),plt.show()

#imgHLS[:,:,1] = ndimage.grey_erosion(imgHLS[:,:,1], size=(1,2));
imgGrad = ndimage.morphology.morphological_gradient(imgHLS[:,:,1], size=(1,4))
#plt.imshow(imgGrad),plt.colorbar(),plt.show()
imgGrad = ndimage.grey_erosion(imgGrad, size=(2,2)); 
#plt.imshow(imgGrad),plt.colorbar(),plt.show()
imgGrad = ndimage.grey_dilation(imgGrad, size=(2,2)); 
#plt.imshow(imgGrad),plt.colorbar(),plt.show()
imgGrad[imgGrad<52] = 0
#plt.imshow(imgGrad),plt.colorbar(),plt.show()

imgHLS[:,:,1] = imgGrad;
imgHLS[imgHLS[:,:,1] == 0] = 0
img = cv2.cvtColor(imgHLS, cv2.COLOR_HLS2RGB )
#plt.imshow(img),plt.colorbar(),plt.show()


labels, count = ndimage.label(imgGrad)
#plt.imshow(labels),plt.colorbar(),plt.show()
#detect colors

#plt.imshow(imgHLS[:,:,0]),plt.colorbar(),plt.show()
#plt.imshow(imgHLS[:,:,1]),plt.colorbar(),plt.show()
#plt.imshow(imgHLS[:,:,2]),plt.colorbar(),plt.show()

colors = {"white":0,"yellow":0, "blue":0, "red":0}




errors = np.zeros((len(notchesTypesExpected.keys()), 4))
i = 0
for imgKey in notchesTypesExpected.keys():
    colors =  {"white":0,"yellow":0, "blue":0, "red":0}
    expectedColors = notchesTypesExpected[imgKey]
    img = imgs[imgKey];
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB )
    imgGrad = ndimage.morphology.morphological_gradient(img[:,:,0], size=(1,2))
    imgGrad[imgGrad<52] = 0
    labels, count = ndimage.measurements.label(imgGrad)
    for label in range(1, count+1):
        labelMeanRGB = np.array(img[labels==label, :].mean(axis=0))
        best_dist = -1
        best_color = ""
        for color in COLOR_VALUES.keys():
            curr_dist = np.sqrt(np.sum((COLOR_VALUES[color] - labelMeanRGB) ** 2))
            if best_dist == -1 or curr_dist < best_dist:
                best_dist = curr_dist
                best_color = color
        print(i,": ",labelMeanRGB," => ", best_color)
        colors[best_color] += 1
    # error detection
    j=0
    for color in colors.keys():
        errors[i, j] += abs(expectedColors[color] - colors[color])
        print(j,"=",color)
        j+=1
    i+=1

sns.heatmap(data=errors)
print("total errors:", errors.sum(axis=0))

#%%

#objective: find the best color codes to diminish errors in their case.
#%%
for i in range(1, count+1):
    labelRGB = originalimg[labels==i, :]
    labelMeanRGB = np.array(labelRGB.mean(axis=0))
    best_dist = -1
    best_color = ""
    print("cluster ",i)
    for color in COLOR_VALUES.keys():
        curr_dist = np.sqrt(np.sum((COLOR_VALUES[color] - labelMeanRGB) ** 2))
        print(color," => ",curr_dist)
        if best_dist == -1 or curr_dist < best_dist:
            best_dist = curr_dist
            best_color = color
    colors[best_color] += 1

print(colors)
#detectedimg = np.zeros(originalimg.shape)
#detectedimg = originalimg[labels != 0] 
#plt.imshow(detectedimg),plt.colorbar(),plt.show()
#showHealthDetected(img)



#%%
labelHLS = imgHLS[labels==1, :]
labelMeanHLS = np.array(labelHLS.mean(axis=0))
labelMeanHLS[1] = labelMeanHLS[1]/256
plt.imshow(labelHLS),plt.colorbar(),plt.show()
plt.imshow([labelMeanHLS]),plt.colorbar(),plt.show()
distToRed = abs(labelMeanHLS[0] - RED_HUE)
distToYellow = abs(labelMeanHLS[0] - ORANGE_HUE)
distToBlue = abs(labelMeanHLS[0] - BLUE_HUE)

distToWhite = abs(labelMeanHLS[1] - 100)

#%%
labelRGB = originalimg[labels==1]

#plt.imshow(labelRGB),plt.colorbar(),plt.show()
labelMeanRGB = np.array([labelRGB.mean(axis=0)])
labelMeanRGB = labelMeanRGB / 255
#plt.imshow(labelMeanRGB),plt.colorbar(),plt.show()
plt.imshow([labelMeanRGB]),plt.colorbar(),plt.show()

#%%
# # reading the current results.


results = pd.read_csv("C:\\Users\\Helldragger\\Documents\\projects\\MetaWatch\\MetaWatch\\data\\temp\\2.csv")

# barlett, kinda clean (4-6 anomalies)
# bohman, blackmanharris, nuttall = same
# rolling window:
# [2.s] 120 clean( 1 anomaly)
# [1.5] 90 semi clean (3 anomalies).
# [1.s] 60 semi clean ( 4 - 6 death anomalies)
# [.5s] 30 ugly (10+)
res2 = results.groupby(['team', "hero"])["health", "armor", "shield", "death"].rolling(120).mean().reset_index()#.unstack(['team', "hero"])

res2["frame"] = res2["level_2"] // 12

res2.loc[res2.death > 0, "death"] = 1 
res2 = res2.drop("level_2", axis=1)
res3 = pd.melt(res2, ['team', "hero", "frame", "death"])

#sns.relplot(x="frame", y="value", hue='variable', col="team", kind="line", data=res3, row="hero")
plt.style.use("seaborn-colorblind")

fig, axes = plt.subplots(6,2, figsize=(1920,1080), dpi=400)


i = 0
for team in res2.team.unique():
    j = 0
    for hero in res2.hero.unique():
        frames = res2.loc[(res2.team==team) & (res2.hero == hero), "frame"]
        health = res2.loc[(res2.team==team) & (res2.hero == hero), "health"]
        shield = res2.loc[(res2.team==team) & (res2.hero == hero), "shield"]
        armor = res2.loc[(res2.team==team) & (res2.hero == hero), "armor"]
        axes[j,i].stackplot(frames, 
                        health, 
                        armor,
                        shield,cmap=plt.get_cmap("Accent"))
        
        j+=1
    i+=1

#plt.title('Recorded Game Statistics')
plt.show()



fig, axes = plt.subplots(6,2)
i = 0
for team in res2.team.unique():
    j = 0
    for hero in res2.hero.unique():
        current_data = res2.loc[(res2.team==team) & (res2.hero == hero)]
        frames = current_data.frame
        daed_frames = (current_data.health < 25) & (current_data.armor < 25) & (current_data.death == 1)
        axes[j,i].stackplot(frames,daed_frames, cmap=plt.get_cmap("Accent"))
        j+=1
    i+=1

#%%

# merging
class DeathInterval:
    def __init__(self, start:int, previous=None, next=None):
        self.end = self.start = start
        self.previous = previous
        self.next = next
    
    def span(self):
        return self.end - self.start
fig, axes = plt.subplots(6,2)
i = 0
for team in res2.team.unique():
    j = 0
    for hero in res2.hero.unique():
        current_data = res2.loc[(res2.team==team) & (res2.hero == hero)]
        frames = current_data.frame
#        daed_frames = (current_data.health < 25) & (current_data.armor < 25) & (current_data.death == 1)
        spawned_frames = (current_data.health >= 25)
        axes[j,i].stackplot(frames,spawned_frames)
        for frame in daed_frames:
            pass # TODO merge small intervals with little distance together, in order to clean the death reports.
        j+=1
    i+=1
        


#%%

# reading ultimate values
import pytesseract
def ocr_core(filename):
    """
    This function will handle the core OCR processing of images.
    """
    text = pytesseract.image_to_string(cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY ),  lang="Koverwatch")  # We'll use Pillow's Image class to open the image and pytesseract to detect the string in the image
    return text

#%%
import pytesseract
path = "C:/Users/Helldragger/Documents/projects/MetaWatch/MetaWatch/src/features/OCRTEST.png"
img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY )
plt.imshow(img)
plt.show()



srcTri = np.array( [[0, 0], [img.shape[1] - 1, 0], [0, img.shape[0] - 1]] ).astype(np.float32)
dstTri = np.array( [[0, 0], [img.shape[1] - 1, 0], [img.shape[1]*0.08, img.shape[0] - 1]] ).astype(np.float32)
warp_mat = cv.getAffineTransform(srcTri, dstTri)
img = cv.warpAffine(img, warp_mat, (img.shape[1], img.shape[0]))

plt.imshow(img)
plt.show()


thresh = 200
maxValue = 255
img = cv2.threshold(img, thresh, maxValue, cv2.THRESH_BINARY )[1]
plt.imshow(img)
plt.show()
text = pytesseract.image_to_string(img, config="digits", lang="Koverwatch")

print("img text: '{}'".format(text))
#%%

import pytesseract

import numpy as np
import cv2 
from matplotlib import pyplot as plt
import seaborn as sns

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
    text = pytesseract.image_to_string(img, config="digits", lang="Koverwatch")

    print("img ult text: '{}'".format(text))
    return text


def test_UltimateParsing(testSources):
    # test format: COLOR_expectedValue.png
    
    expected = {}
    testedColors = set();
    testedValues = set();
    colorToIndex = {}
    valueToIndex = {}
    colorLabels = []
    valueLabels = []
    for src in testSources:
        color, value  = src.split("_");
        expected[src] = {"value":value, "color":color};
        if color not in testedColors:
            colorToIndex[color] = len(testedColors)
            testedColors.add(color)
            colorLabels.append(color)
        if value not in testedValues:
            valueToIndex[value] = len(testedValues)
            testedValues.add(value)
            valueLabels.append(value)


    imgs = {}
        
    errors = np.zeros((len(testedValues),  len(testedColors)))
    i = 0
    legendPrinted = False
    for src in expected.keys():
        ultimateImg = cv2.imread('src/data/tests/Ultimate/'+src+'.png',-1);
        ultimateImg = cv2.cvtColor(ultimateImg, cv2.COLOR_BGR2RGB )
        assert ultimateImg is not None
        
        value = convertUltimateImgToText(ultimateImg)
                   
        if value != expected[src]["value"]:
            errors[valueToIndex[expected[src]["value"]], colorToIndex[expected[src]["color"]]] += 1
    

    sns.heatmap(data=errors, xticklabels=colorLabels, yticklabels=valueLabels)#.get_figure()#.savefig("src/data/tests/Ultimate/Tests_ErrorHeatmap.png")
    plt.show()
    totalErrors = errors.sum(axis=0).sum()
    print("total color errors:", errors.sum(axis=0))
    print("total values errors:", errors.sum(axis=1))
    print("total errors:", errors.sum(axis=0).sum())

    ultimateAcc = 1 - (totalErrors / len(testSources))
    print("Ultimate detection accuracy: ", ultimateAcc)
    #assert ultimateAcc >= 0.90 # 10% d'erreur



FULL_TEST_SOURCES = [
        "BLUE_0",
        "BLUE_1",
        "BLUE_14",
        "BLUE_15",
        "BLUE_16",
        "BLUE_24",
        "BLUE_25",
        "GREEN_13",
        "GREEN_16",
        "GREEN_21",
        "GREEN_22",
        "GREEN_39",
        "GREEN_42",
        "GREY_0",
        "GREY_11",
        "GREY_13",
        "GREY_34",
        "ORANGE_0",
        "RED_10",
        "RED_26",
        "WHITE_0",
        "WHITE_1",
        "WHITE_8",
        "WHITE_21",
        "WHITE_24",
        "WHITE_34"
    ]

ACTUAL_TEST_SOURCES = [
    "BLUE_0",
    "BLUE_1",
    "BLUE_14",
    "BLUE_15",
    "BLUE_16",
    "BLUE_24",
    "BLUE_25",
]

test_UltimateParsing(ACTUAL_TEST_SOURCES)
#%%
from os import listdir
from os.path import isfile, join
imgDir = "src/data/tests/Ultimate/"
onlyfiles = [f for f in listdir(imgDir) if isfile(join(imgDir, f))]


imgHeatmap = None;
lastImg = None;
for fileSrc in onlyfiles:
    img = cv.imread(imgDir+"/"+fileSrc)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY);
    if imgHeatmap is None:
        imgHeatmap=np.zeros(img.shape)
        lastImg = img
    
    if img.shape != imgHeatmap.shape:
        continue
    else:
        imgHeatmap = imgHeatmap + img

sns.heatmap(imgHeatmap);
plt.show();
thresholedHeatmap = imgHeatmap
thresholedHeatmap[thresholedHeatmap> 110000] = 0
sns.heatmap(thresholedHeatmap);

plt.show();
#%%
plt.show();
cv.imshow("heatmap",imgHeatmap)
cv.waitKey(0)