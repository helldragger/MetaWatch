
#%%
from os import listdir
from os.path import isfile, join
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



#%%
imgDir = "src/data/tests/Team/"
imgs = {}
srcToIndex = {}
srcIndex = []
for f in listdir(imgDir) :
    if isfile(join(imgDir, f)):
        src = join(imgDir, f)
        imgs[f] = cv.cvtColor(cv.imread(src), cv.COLOR_BGR2RGB)
        srcToIndex[f] = len(srcIndex)
        srcIndex.append(f)



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

import pytesseract
def parseTeam(img): 
    img = apply_brightness_contrast(img, contrast=20);
    text = pytesseract.image_to_string(img, config="-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ -c preserve_interword_spaces=1 --psm 7")
    return text, img

errors = np.zeros((len(imgs),1))

for src in imgs.keys():
    expected, version = src.split(" ")
    expected = expected.replace("_", "")
    result, resultImg = parseTeam(imgs[src])
    if result != expected:
        #plt.imshow(resultImg);
        #plt.show();
        print("ERR:  Expected '{}'; Parsed '{}'".format(expected, result))
        errors[srcToIndex[src],0] = 1

sns.heatmap(data=errors, yticklabels=srcIndex, xticklabels=["Errors"])

totalErrors =  errors.sum(axis=0)[0]
print("Total errors: ", totalErrors)
accuracy = (1 - (totalErrors / len(imgs)))*100
print("Accuracy: ", accuracy,"%")
