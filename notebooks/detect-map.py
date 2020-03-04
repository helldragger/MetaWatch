#%%


import cv2
import numpy as np;

from scipy import ndimage

from scipy import spatial
from scipy.spatial.distance import cdist


# Map detection:
# 
# 1) use a color hhistogram to determine the color code of the current nation.

# 2) use gfeature matching to detect specific features for each submap of the nation.


