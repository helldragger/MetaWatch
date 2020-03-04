#%%
import cv2
import numpy as np;

from scipy import ndimage

from scipy import spatial
from scipy.spatial.distance import cdist

def convertHeroHealthImgToNotches(healthImg):
    img = cv2.cvtColor(healthImg, cv2.COLOR_BGR2RGB )
    imgGrad = ndimage.morphology.morphological_gradient(img[:,:,0], size=(1,2))
    # see feature building experiments
    imgGrad[imgGrad<52] = 0
    return ndimage.measurements.label(imgGrad)

def convertColorsToData(colors):
    return {"health":colors["white"]*25,
            "armor":colors["yellow"]*25,
            "shield":colors["blue"]*25,
            "death":colors["red"] };

def convertNotchesToColors(notches, img):
    # notches = (notches pixels, amount)
    # for now we will focus on the amount
    colors = {"white":0,"yellow":0, "blue":0, "red":0}

    COLOR_VALUES = {
        "white":np.array( [160, 160, 160]),
        "yellow":np.array([208, 140, 127]), 
        "blue":np.array([66, 167, 194]),
        "red":np.array([200, 15, 36]),
    }
    labels, count = notches
    for label in range(1, count+1):
        labelMeanRGB = np.array(img[labels==label, :].mean(axis=0))
        best_dist = -1
        best_color = ""
        for color in COLOR_VALUES.keys():
            curr_dist = np.sqrt(np.sum((COLOR_VALUES[color] - labelMeanRGB) ** 2))
            if best_dist == -1 or curr_dist < best_dist:
                best_dist = curr_dist
                best_color = color
        colors[best_color] += 1

    return colors

#%%


def convertNotchesToColors2(notches, img):
    # notches = (notches pixels, amount)
    # for now we will focus on the amount
    colors = {"white":0,"yellow":0, "blue":0, "red":0}

    indexToColor = ["white", "yellow", "blue", "red"]
    COLORS = np.array([
        [160, 160, 160],# white
        [208, 140, 127],# yellow
        [66, 167, 194], # blue
        [200, 15, 36]]) # red

    labels, count = notches
    for label in range(1, count+1):
        labelMeanRGB = np.array(img[labels==label, :].mean(axis=0))
        dist = cdist( COLORS, [labelMeanRGB], metric="euclidean" )  # -> (nx, ny) distances
        best_color  = indexToColor[np.argmin(dist)]
        colors[best_color] += 1

    return colors




img = cv2.imread('src/data/tests/Health/frame377_hero_A1_health.jpg',-1);
notches = convertHeroHealthImgToNotches(img)
colors1 = convertNotchesToColors(notches, img)
colors2 = convertNotchesToColors2(notches, img)
assert colors1 == colors2
