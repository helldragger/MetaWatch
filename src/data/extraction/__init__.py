
from matplotlib import pyplot as plt
import cv2
import numpy as np;

from scipy import ndimage
import pickle

from scipy import spatial
from scipy.spatial.distance import cdist
from typing import *
import math
import pytesseract

from os import listdir
from os.path import isfile, join

from contextlib import ExitStack

from logging import FileHandler
from vlogging import VisualRecord

from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color

import logging

MATCH_COLOR = (100,100,255)
MATCH_WEIGHT = 1

def logMatchesOnBGRImg(img_BGR, matches, title="Matches", text="", step=""):
	img = img_BGR.copy()
	for match in matches:
		cv2.rectangle(img,(match[0],match[1]),(match[2],match[3]),MATCH_COLOR,MATCH_WEIGHT)
	logging.getLogger("killfeed").debug(VisualRecord(title+step, img, text, fmt="png"))
	
def logMatchesOnGrayImg(img_Gray, matches, title="Matches", text="", step=""):
	logMatchesOnBGRImg(cv2.cvtColor(img_Gray, cv2.COLOR_GRAY2BGR), matches, title, text, step)

def logMatchesOnImg(img, matches, title="Matches", text=""):
	if len(img.shape) == 2:
		logMatchesOnGrayImg(img, matches, title, text)
	elif len(img.shape) == 3:
		logMatchesOnBGRImg(img, matches, title,text)

def logMatchOnImg(img, match, title="Match", text=""):
	logMatchesOnImg(img, [match], title, text)

def logScaledTemplateMatchOnBGRImg(img_BGR, template, loc, scale, title="scaled template match (template, match)", text=""):
	result_img = img_BGR.copy()
	newH, newW = (math.ceil(template.shape[0] * scale), math.ceil(template.shape[1] * scale))

	cv2.rectangle(result_img,(loc[0],loc[1]),(loc[0]+newW,loc[1]+newH),MATCH_COLOR,MATCH_WEIGHT)
	
	logging.getLogger("killfeed").debug(VisualRecord(title, [template, result_img], text, fmt="png"))

def logScaledTemplateMatchOnGrayImg(img_Gray, template, loc, scale, title="scaled template match (template, match)", text=""):
	img_BGR = cv2.cvtColor(img_Gray, cv2.COLOR_GRAY2BGR)
	logScaledTemplateMatchOnBGRImg(img_BGR, template, loc, scale, title, text)

def logScaledTemplateMatch(img, template, loc, scale, title="scaled template match (template, match)", text=""):
	if len(img.shape) == 2:
		logScaledTemplateMatchOnGrayImg(img, template, loc, scale, title, text)
	elif len(img.shape) == 3:
		logScaledTemplateMatchOnBGRImg(img, template, loc, scale, title, text)

def logCroppedTemplateMatchOnBGRImg(img_BGR, template, loc, cropType, title="scaled template match (template, match)", text=""):
	result_img = img_BGR.copy()
	cropped_template = cropIcon(template, cropType)

	newH, newW = cropped_template.shape[:2]

	cv2.rectangle(result_img,(loc[0],loc[1]),(loc[0]+newW,loc[1]+newH),MATCH_COLOR,MATCH_WEIGHT)
	
	logging.getLogger("killfeed").debug(VisualRecord(title, [template, result_img], text, fmt="png"))

def logCroppedTemplateMatchOnGrayImg(img_Gray, template, loc, cropType, title="scaled template match (template, match)", text=""):
	img_BGR = cv2.cvtColor(img_Gray, cv2.COLOR_GRAY2BGR)
	logCroppedTemplateMatchOnBGRImg(img_BGR, template, loc, cropType, title, text)

def logCroppedTemplateMatch(img, template, loc, cropType, title="scaled template match (template, match)", text=""):
	if len(img.shape) == 2:
		logCroppedTemplateMatchOnGrayImg(img, template, loc, cropType, title, text)
	elif len(img.shape) == 3:
		logCroppedTemplateMatchOnGrayImg(img, template, loc, cropType, title, text)

def logImg(img, title="", text=""):
	logging.getLogger("killfeed").debug(VisualRecord(title,img,text, fmt="png"))


def convertHeroHealthImgToNotches(img)-> Tuple[object, int]:
	#img = cv2.cvtColor(healthImg, cv2.COLOR_BGR2RGB )
	imgGrad = ndimage.morphology.morphological_gradient(img[:,:,0], size=(1,2))
	# see feature building experiments
	imgGrad[imgGrad<52] = 0
	return ndimage.measurements.label(imgGrad)

def convertColorsToData(colors:Dict[str, int])-> Dict[str, int]:
	return {"health":colors["white"]*25,
			"armor":colors["yellow"]*25,
			"shield":colors["blue"]*25,
			"damage":colors["red"] };

def convertNotchesToColors(notches:Tuple[object, int], img) -> Dict[str, int]:
	# notches = (notches pixels, amount)
	# for now we will focus on the amount
	colors = {"white":0,"yellow":0, "blue":0, "red":0}

	indexToColor = ["white", "yellow", "blue", "red"]
	COLORS = np.array([
		[160, 160, 160],# white in BGR
		[127, 140, 208],# yellow
		[194, 167, 66], # blue
		[36, 15, 200]]) # red

	labels, count = notches
	for label in range(1, count+1):
		labelMeanRGB = np.array(img[labels==label, :].mean(axis=0))
		dist = cdist( COLORS, [labelMeanRGB], metric="euclidean" )  # -> (nx, ny) distances
		best_color  = indexToColor[np.argmin(dist)]
		colors[best_color] += 1

	return colors


def cropImage(img, cropType:Tuple[str, int, str], cropMask, cache:Dict[Tuple[str, int, str], Tuple[int,int,int,int]]={}):
	if cache.get(cropType, None) is None:
		y,x = cropMask[:,:,0].nonzero() # get the nonzero alpha coordinates
		minx = np.min(x)
		miny = np.min(y)
		maxx = np.max(x)
		maxy = np.max(y) 
		cache[cropType] = (miny, maxy, minx, maxx)
	miny, maxy, minx, maxx = cache[cropType]
	return img[miny:maxy, minx:maxx]
			
def frameToDate(frame:int, FPS:int=60) -> str:
	sign = "" if frame > 0 else "-"
	#assuming 60 fps
	seconds = frame / FPS
	minutes = seconds // 60
	hours = minutes // 60
	return sign + f'{int(hours % 24):02}:{int(minutes % 60):02}:{int(seconds % 60):02}' 
		
def dateToFrame(date:int, FPS:int) -> str:
	sign = -1 if date.startswith("-") else 1
	if sign == -1:
		date = date[1::]
	# assuming 60 fps
	# formats: hh:mm:ss, hh and mm might be missing, hence, we start the processing while reversing the elements
	date_values = date.split(":")
	elem_i = 1
	frame = 0
	for value in date_values[::-1]: # as such: ss, then mm, then hh
		frame += int(value) * (FPS**elem_i) # as secs, elem is 1, the increment is 60 per sec. at mins, it's 60*60, hence 60**2, and 60**3 for hours
		elem_i+=1

	return sign * frame



def convertUltimateImgToValue(img)->str:
	# we consider 80% of the reference images must agree to consider a detection succesful.
	MIN_DETECTION_COUNT = math.floor(len(UltimateFeatureCache) * .8)

	kp, des = kaze.detectAndCompute(img, None)
	if des is None or kp is None:
		# can happen if the image contains nothing, i.e. during a black screen
		return False
	if len(kp) < 3 or len(des) < 3 :
		# can happen if the image contains too few data, i.e. during a black screen
		return False
	value = 0
	for TRUTH_src, TRUTH_kp, TRUTH_des in UltimateFeatureCache:
		matches = flann.knnMatch(TRUTH_des,des,k=2)
		MIN_MATCH_COUNT = math.floor(len(TRUTH_kp)*.8) # 70% des features de l'image de reference doivent coller. 
		# store all the good matches as per Lowe's ratio test.
		good = []
		for m,n in matches:
			if m.distance < 0.7*n.distance:
				good.append(m)
		# here we add 1 to value if the current reference img detects a ultimate
		value+= 1 if len(good)>=MIN_MATCH_COUNT else 0
	# here we consider the detection valid if enough reference images detected it
	return value>=MIN_DETECTION_COUNT


def apply_brightness_contrast(input_img, brightness:int = 0, contrast:int = 0):

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


def loadVideo(input_filepath:str):
	return cv2.VideoCapture(input_filepath+"/replay.mp4");

def loadMaskImg(mask_path:str):
	img = cv2.imread(mask_path,-1)
	#TODO detect and throw an error when non binary img detected.
	return img

def loadMaskCache(input_filepath:str):
	maskCache = {}
	for team in ["A", "B"]:
		# Thresholding allows using grey scale imgs, when above 50/255 (25%) white it is considered white, otherwise black
		maskCache[(team, "Team")] = loadMaskImg(input_filepath+"/masks/Team/"+team+".png");

		for hero in range(1,7,1):
			for maskType in ["Health", "Nickname", "Ultimate", "Hero"]:
				maskCache[(team, hero, maskType)] = loadMaskImg(input_filepath+"/masks/"+maskType+"/"+team+str(hero)+".png");
	for uiMask in ["killfeed"]:
	   maskCache[("UI", uiMask)] = loadMaskImg(input_filepath+"/masks/UI/"+uiMask+".png"); 

	return maskCache

def loadUltimateFeatureCache():
	featureCache = []
	imgDir = "assets/tests/Ultimate/"
	onlyfiles = [f for f in listdir(imgDir) if (isfile(join(imgDir, f)) and f.split(" ")[0]=="100")]

	for src in onlyfiles:
		TRUTH_img = cv2.imread(join(imgDir, src),-1);
		TRUTH_kp, TRUTH_des = kaze.detectAndCompute(TRUTH_img,None)
		featureCache.append((src, TRUTH_kp, TRUTH_des))
   
	return featureCache

def convertImgToText(img):
	img = apply_brightness_contrast(img, contrast=20);
	text = pytesseract.image_to_string(img, config="-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 6")
	return text

def convertImgToTeamname(img): 
	img = apply_brightness_contrast(img, contrast=20);
	text = pytesseract.image_to_string(img, config="-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ -c preserve_interword_spaces=1 --psm 7")
	return text

def hasBeenEvaluatedAlready(match, evaluated_matches):
	TOLERANCE = 30 # en pixels de distance
	#print("match:", match)
	#print("evaluated:",evaluated_matches)
	abs_diff = np.abs( evaluated_matches - match )
	#print("abs diff:",abs_diff)
	sum_diff = np.sum( abs_diff , axis=0 if len(abs_diff.shape) == 1 else 1)
	#print("sum diff:",sum_diff)
	min_sum  = np.min( sum_diff )
	#print("min sum:",min_sum)
	return min_sum < TOLERANCE 


def uniqueMatches(matches):
	evaluated_matches = None
	# si on as deja evalué un autre match très très similaire à notre match actuel 
	# alors on l'ignore, c est surement le même
	
	first = True
	for match in matches:
		match = np.array(match)
		if first :
			first = False
			evaluated_matches = match
			continue

		if not hasBeenEvaluatedAlready(match, evaluated_matches):
			evaluated_matches= np.vstack((evaluated_matches, match))
	if evaluated_matches is None:
		return []
	else:
		if len(evaluated_matches.shape) == 1:
			# if evaluated matches contains only one dimension, it's a single match
			return [evaluated_matches]
		else:
			# else it s a list of matches
			return list(evaluated_matches)

def getTemplateMatches(img, template, show=False, threshold=0.8, tolerance=0.7):
	# can be upgraded by adding the matches probabilities in the output
	w, h = template.shape[::-1]

	if w > img.shape[1] or h > img.shape[0]:
		raise ValueError("template bigger than the img")

	logImg(template, "matching: template")
	logImg(img, "matching: matched image")
	
	res = cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
	#res[res < max(threshold, np.max(res)*tolerance)] = 0
	#if show:
		#cv2.imshow("template", template)
		#cv2.imshow("img", img)
		#print("min",np.min(res),"max", np.max(res),"mean", np.mean(res))
		#cv2.imshow("res",res)
		#res_threshold = res
		#res_tolerance = res

		#res_threshold[res_threshold < threshold] = 0
		#res_tolerance[res_tolerance < np.max(res_tolerance)*tolerance] = 0
		
		#cv2.imshow("res, threshold",res_threshold)
		#cv2.imshow("res, tolerance",res_tolerance)

		#print(len(np.where (res >= threshold)))
		#print(len(np.where (res >= np.max(res)*tolerance)))
		#cv2.waitKey(0)
	logImg((res*125)+125, "match results", "highest : "+str(np.max(res))+"; lowest : "+str(np.min(res)))
	if np.min(res) == 1.0: # if there is no actual match, everything is at 1.0, so we just say no matches.
		return []

	
	# get each loc where value is good
	loc = np.where (res >= max(threshold, np.max(res)*tolerance))
	#print(loc)
	# get each loc value
	loc_values = res[ loc ]
	#print(loc_values)
	# map loc to its value
	loc_pairs = list(zip(*loc, loc_values))
	#print(loc_pairs)
	# sort by value
	sorted_locs = sorted(loc_pairs, key=lambda loc_data: loc_data[2])
	#print(sorted_locs)
	# 10 best values only
	best_locs = sorted_locs[-100:]
	# sort locs by vertical value, to retrieve the previous sorting system
	best_locs = sorted(best_locs, key=lambda loc_data: loc_data[0])
	#print(best_locs)
	# [(x, y, v),()] -> [(x, y),()] -> np.array[[],[]]
	matches_rects = np.array(list(map(lambda loc_data:(loc_data[1], loc_data[0], loc_data[1]+w, loc_data[0]+h ), best_locs)))
	#print(best_locs)
	#print(len(loc), "matches detected")
	#loc = np.where( res >= threshold)
	# [[x,y], []] -> [ [x, y, x+w, y+h],[] ]
	# we keep only the unique matches (+- tolerance)
	return uniqueMatches(matches_rects)


def getBestScaleInvariantTemplateMatch(img, template):
	(iH, iW) = img.shape[:2]
	(tH, tW) = template.shape[:2]
	template_TBResized = template.copy()

	found = None
	# to avoid the problem of having the template at a different scale than the image,
	# and as such not recognizing the template, we test multiple scales to check for the best possible matching

	for scale in np.linspace(0.5, 1.5, 5): 
		newW, newH = (math.ceil(template_TBResized.shape[0] * scale), math.ceil(template_TBResized.shape[1] * scale))
	 
		if newH > iH or newW > iW:
			continue

		template_resized = cv2.resize(template_TBResized, (newH, newW))
 
		result = cv2.matchTemplate(img, template_resized, cv2.TM_CCOEFF_NORMED)
		(_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
		

		if found is None or maxVal > found[0]:
			found = (maxVal, maxLoc, scale)
	
	
	if found is None:
		return None
	else:
		return (found[1][0],found[1][1],found[1][0]+tW,found[1][1]+tH), found[0], found[2]

def cropImg(img, scale, cropType):
	(H, W) = img.shape[:2]
	#logImg(img, "pre crop img", "crop type: "+str(cropType))
	img_cropped = img.copy()
	newW, newH = (math.ceil(img.shape[0] * scale), math.ceil(img.shape[1] * scale))

	marginH = math.ceil((H-newH)/2)
	marginW = math.ceil((W-newW)/2)
	if cropType==0: # both
		img_cropped = img_cropped[marginH:-marginH,marginW:-marginW] 
	elif cropType == 1: #vertical
		img_cropped = img_cropped[marginH:-marginH,:] 
	elif cropType == 2: #horizontal
		img_cropped = img_cropped[:,marginW:-marginW] 
	else:
		raise ValueError("croptype can only be 0 (both), 1 (vertical) or 2 (horizontal)")
	
	#logImg(img_cropped, "post crop img", "crop type: "+str(cropType))
	return img_cropped


def cropImgMultiScale(img, scaleW, scaleH):
	(H, W) = img.shape[:2]
	#logImg(img, "pre crop img", "crop type: W "+str(scaleW)+ " H "+str(scaleH) )
	newH, newW = (math.ceil(H * scaleH), math.ceil(W * scaleW))
	
	marginH = math.ceil((H-newH)/2)
	marginW = math.ceil((W-newW)/2)
	if marginH == 0 and marginW == 0:
		return img[:,:]
	elif marginH == 0:
		return img[:,marginW:-marginW]
	elif marginW == 0:
		return img[marginH:-marginH,:]
	else:
		return img[marginH:-marginH,marginW:-marginW] 


def cropIconToLong(img):
	# when long, 
	# the image is cropped accordingly
	# then the image is scaled down to the appropriate resolution
	cropped_img = img[:,:]
	cropped_img = cropImgMultiScale(cropped_img, 1, 0.65)
	return cv2.resize(cropped_img, (int(img.shape[1] * 0.85), int(img.shape[0] * 0.55) ))


def cropIconToTall(img):
	cropped_img = img[:,:]
	cropped_img = cropImgMultiScale(cropped_img, 0.8, 1)
	cropped_img = cropped_img[:, :-int(cropped_img.shape[1] * 0.3)]
	return cv2.resize(cropped_img, (int(img.shape[1] * 0.47), int(img.shape[0] * 0.87) ))


def cropIconToFlat(img):
	# when flat
	# then, the image is cropped accordingly
	# then the image is scaled down to the appropriate resolution
	cropped_img = img[:,:]
	cropped_img = cropImgMultiScale(cropped_img, 0.9, 0.9)
	cropped_img = cropped_img[:-int(cropped_img.shape[0] * 0.2),:]
	return cv2.resize(cropped_img, (int(img.shape[1] * 0.6), int(img.shape[0] * 0.46) ))

def cropIcon(img, cropType):
	if cropType == "LONG":
		return cropIconToLong(img)
	elif cropType == "TALL":
		return cropIconToTall(img)
	elif cropType == "FLAT":
		return cropIconToFlat(img)
	else:
		return img

def getBestCroppedInvariantTemplateMatch(img, template, CROPTYPE_WHITELIST=["NORMAL", "LONG"]):
	(iH, iW) = img.shape[:2]
	(tH, tW) = template.shape[:2]
	template_TBCropped = template.copy()
	
	found = None
	# to avoid the problem of having the template at a different scale than the image,
	# and as such not recognizing the template, we test multiple scales to check for the best possible matching

	for cropType in CROPTYPE_WHITELIST:

		template_cropped = cropIcon(template_TBCropped, cropType)
		#if cropType == "TALL":
		#	logImg(template_cropped, "TALL template")

		(newH, newW) = template_cropped.shape[:2]
		if newH > iH or newW > iW:
			# we don't try if the normal one cannot fit
			break
		
		result = cv2.matchTemplate(img, template_cropped, cv2.TM_CCOEFF_NORMED)
		(_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
		match = (maxLoc[0], maxLoc[1],maxLoc[0]+newW, maxLoc[1]+newH)

		#if cropType == "TALL":
			#logImg(template_cropped, "FLAT template")
		#	logMatchOnImg(img, match, "TALL template match", maxVal)
		if found is None or maxVal > found[1]:
			found = (match, maxVal, cropType)

	
	
	if found is None:
		return None
	else:
		return found

def isCroppedIcon(img, template, CROPTYPE_WHITELIST):
	if "NORMAL" not in CROPTYPE_WHITELIST:
		return True
	# it's cropped if any cropped value is > 0.9 (almost certain)
	results = []
	for cropType in CROPTYPE_WHITELIST:
		if cropType == "NORMAL":
			continue
		else:
			results.append(cv2.minMaxLoc(cv2.matchTemplate(img, cropIcon(template, cropType), cv2.TM_CCOEFF_NORMED))[1])
	return max( results ) > 0.8

def getBestTemplateMatch(img, template):
	w, h = template.shape[::-1]
	if w > img.shape[1] or h > img.shape[0]:
		raise ValueError("template bigger than the img")
	
	result = cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
	_minVal, _maxVal, minLoc, maxLoc = cv2.minMaxLoc(result, None)
	#result_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	
	
	#cv2.rectangle(result_img,(maxLoc[0],maxLoc[1]),(maxLoc[0]+w,maxLoc[1]+h),(100,100,255),3)
	#text =  "val = " +str(round(_maxVal, 3))
	
	#logging.getLogger("killfeed").debug(VisualRecord("Best template match (template, match)", [template, result_img], text, fmt="png"))


	return (maxLoc[0],maxLoc[1],maxLoc[0]+w,maxLoc[1]+h), _maxVal
	

def printTemplateMatches(img, matches):
	for rect in matches:
		cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0,0,255), 2)
	cv2.imshow("matches",img)
	cv2.waitKey(0)


def getAbilityImg(hero_name, ability_name):
	img_src = abilities_src[hero_name][ability_name]
	ability_img = cv2.imread(img_src, 0 )
	# arbitrary value, a cropping happens at approx 42 pixels, 
	# the abilities masks are 46 pixels squares, the kill arrow is 26px high, 
	# and the abilities are approx 1.5* this height

	return ability_img 



# Detection of heroes:
# We try to match every heroes known on the killfeed line, we keep the matches and the position relative to the arrow will determine if it's a kill or a killer
def detectHero(cropped_img, isLong=None):
	best_hero = None
	best_value = 0.78
	
	cropped_hero_matches = {}
	for hero_name in icons_src.keys():
		hero_img = cv2.imread(icons_src[hero_name], 0)
		try:
			if isLong is None:
				WHITELIST = ["NORMAL","LONG"]
			elif isLong:
				WHITELIST = ["LONG"]
			else:
				WHITELIST = ["NORMAL"]

			result = getBestCroppedInvariantTemplateMatch(cropped_img, hero_img, CROPTYPE_WHITELIST=WHITELIST)
			
			if result is not None:
				match, value, cropType = result
				if value > best_value: #detection threshold
					#logCroppedTemplateMatch(cropped_img, hero_img, match, cropType, title="HERO UPDATED:"+hero_name, text="value @"+str(value))
					best_hero = (hero_name, match, value, cropType)
					best_value = value
				else:
					#logCroppedTemplateMatch(cropped_img, hero_img, match, cropType, title="HERO IGNORED: "+hero_name, text="value @"+str(value))
					pass
			
			if best_value > 0.9:
				break;
		except ValueError as err:
			# not enough space to fit the template, so we call it a no hero found.
			
			#print("ERROR > not enough space to search for",hero_name,"hero template.")
			#print("img shape:",cropped_img.shape, "template shape:",hero_img.shape)
			continue
	#print("hero detected: ", best_hero)
	#logMatchOnImg(cropped_img, best_hero[1], title="HERO DETECTED:"+best_hero[0], text="value @"+str(best_hero[2])+"; type: "+best_hero[3]+" @ coords "+str(best_hero[1]))
	return best_hero

# we do the same with any assists
def detectAssists(cropped_img, threshold=0.7, isLong=None):
	assists = []
	assists_matches = []

	for hero_name in icons_src.keys():
		hero_img = cv2.imread(icons_src[hero_name], 0)
		try:

			if isLong is None:
				WHITELIST = ["TALL", "FLAT"]
			elif isLong:
				WHITELIST = ["FLAT"]
			else:
				WHITELIST = ["TALL"]

			result = getBestCroppedInvariantTemplateMatch(cropped_img, hero_img, CROPTYPE_WHITELIST=WHITELIST)
			if result is not None:
				match, value, cropType = result
				if value > threshold: #detection threshold
					#logCroppedTemplateMatch(cropped_img, hero_img, match, cropType, title="ASSIST DETECTED: "+hero_name, text="value @"+str(value))
					assists.append(hero_name)
					assists_matches.append(match)
				else:
					pass
					#logCroppedTemplateMatch(cropped_img, hero_img, match, cropType, title="ASSIST IGNORED: "+hero_name, text="value @"+str(value))

		except ValueError as err:
			# impossible to fit any template in the assis zone, so no assists.
			#print("Impossible to fit", hero_name, "template into the assist img")
			continue

	return assists, assists_matches


# Feature matching for abilities: not usable.

def detectAbility(cropped_img, hero_name):
	current_detected_ability = "none"
	best_ability_ratio = 0

	if abilities_src.get(hero_name, None) is None:
		# no abilities registered for this hero
		return current_detected_ability

	if min(cropped_img.shape) < 43:
		# there is not enough space for any ability here, so there was none used.
		return current_detected_ability

	if len(abilities_src[hero_name].keys()) == 1:
		# we can consider there is actually something here.
		return  list(abilities_src[hero_name].keys())[0]

	THRESHOLD = 200

	cropped_img[cropped_img < THRESHOLD] = 0
	cropped_img[cropped_img >= THRESHOLD] = 255
	cropped_img_canny = cv2.Canny(cropped_img, 100, 200)
	for ability_name in abilities_src[hero_name].keys():
		ability_img = getAbilityImg(hero_name, ability_name)
		ability_img_canny = cv2.Canny(ability_img, 100, 200)
		result = getBestScaleInvariantTemplateMatch(cropped_img_canny, ability_img_canny)
		if result is not None:
			ability_match, ability_ratio, _ = result
			if ability_ratio > best_ability_ratio:
				best_ability_ratio = ability_ratio
				current_detected_ability = ability_name
	return current_detected_ability



def cropAroundKillArrow(img, match):
	h = match[3] - match[1]
	minh = max(math.floor(match[1]-(.3*h)), 0)
	maxh = min(math.ceil(match[3]+(.3*h)), img.shape[0])
	killers_img, killed_img = img[minh:maxh, -600:match[0]], img[minh:maxh, match[2]:]
	
	#logMatchOnImg(img, match, "Cropping around match : before")
	#logImg(killers_img, "left crop")
	#logImg(killed_img, "right crop")
	return killers_img, killed_img


def cropAroundMatch(img, match):
	left_img, right_img = img[:, :match[0]], img[:, match[2]:]
	#logMatchOnImg(img, match, "Cropping around match : before")
	#logImg(left_img, "left crop")
	#logImg(right_img, "right crop")
	return left_img, right_img


def is_kill_arrow_white(img_BGR, killArrow_img, match):
	mask = killArrow_img
	mask[mask < 90] = 0
	mask[mask >= 90] = 1

	cropped_img = img_BGR[match[1]:match[3], match[0]:match[2], :]
	masked_img = cropped_img;
	masked_img[mask] = 0

	summed_color = np.sum(np.sum(masked_img, axis=0), axis=0)
	normalized_color = summed_color / np.max(summed_color)
	# if it's white, then the normalized color should have three values next to 1, if a color was dominant, only one of them would be next to 1.
	max_diff = np.max(normalized_color) - np.min(normalized_color)
	dominant_color = np.argmax(normalized_color)
	# if the colors are approximately equivalent, then it's white.
	return max_diff < .5 # white

def sampleColor(img, coord, title="Color sample"):
	maxY, maxX, _ = img.shape
	color = img[coord[0],coord[1],:]
	padding=2
	match = (max(0, coord[1]-padding), max(0, coord[0]-padding), min(maxX, coord[1]+padding), min(maxY, coord[0]+padding))
	logMatchOnImg(img,match,title, "value: "+str(color)+" @ "+str(coord))
	return color


def detectKilledTeamColor(killed_name_img_BGR):
	maxY, maxX, _ = killed_name_img_BGR.shape
	# we center on the vertical axis to be in th emiddle of the kill feed ticket
	# we then offset just a little after the hero split to get the color between the hero frame and the player nickname 
	X = 6
	
	if X < maxX :
		coord = (int(maxY*0.5),X)
		return sampleColor(killed_name_img_BGR, coord, title="Killed team color sample")
	return None



def detectKillerTeamColor(killer_name_img_BGR):
	maxY, maxX, _ = killer_name_img_BGR.shape
	# we center on the vertical axis to be in th emiddle of the kill feed ticket
	# we then offset just a little after the hero split to get the color between the hero frame and the player nickname 
	X = 6
	if X < maxX :
		coord = (int(maxY*0.5), maxX-X)
		return sampleColor(killer_name_img_BGR, coord, title="Killer team color sample")
	return None

# see example here http://hanzratech.in/2015/01/16/color-difference-between-2-colors-using-python.html
from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color

def convertColorToTeam(unknown_color, team_A_color, team_B_color):
	unknown_color = sRGBColor(unknown_color[2]/255, unknown_color[1]/255, unknown_color[0]/255)
	unknown_color = convert_color(unknown_color, LabColor);
	dist_A = delta_e_cie2000(unknown_color, team_A_color);
	dist_B = delta_e_cie2000(unknown_color, team_B_color);
	if dist_A < dist_B:
		return "A"
	else:
		return "B"


def detectKillArrow(img_BGR,killArrow_img):
	img_gray = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2LAB)[:,:,0]
	#killArrow_canny = cv2.Canny(killArrow_img,100,200)

	#we remove the leftmost edge, to allow the detection of arrows combined to other abilities images
	#killArrow_canny_noedge = killArrow_canny[3:,:]
	
	#img_canny = cv2.Canny(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2LAB)[:,:,0],100,200)
	
	#edge_matches = getTemplateMatches(img_canny, killArrow_canny, show=False, threshold=0.8)
	#logMatchesOnGrayImg(img_canny, edge_matches, "edge only kill arrow matches")
	#print(edge_matches)

	#noedge_matches = getTemplateMatches(img_canny, killArrow_canny_noedge, show=False, threshold=0.8)
	#logMatchesOnGrayImg(img_canny, noedge_matches, "no edge only kill arrow matches")
	
	full_matches = getTemplateMatches(img_gray, killArrow_img, show=False, threshold=0.76)
	#logMatchesOnBGRImg(img_BGR, full_matches, "full kill arrow matches")
	#print(noedge_matches)
	#all_matches = edge_matches
	#all_matches.extend(noedge_matches)
	#all_matches.extend(full_matches)
	#print(all_matches)
	#unique_matches = uniqueMatches(all_matches)
	unique_matches = uniqueMatches(full_matches)
	#print(unique_matches)

	final_matches = sorted(unique_matches, key=lambda loc_data:loc_data[1])
	#print(final_matches)
	logMatchesOnBGRImg(img_BGR, final_matches, "final kill arrow matches")
	return final_matches

def readKillFeed(img_BGR, killArrow_img, team_A_color, team_B_color):
	#logImg(img_BGR, title="Extracting killfeed from this image")
	#logImg(killArrow_img, title="Killarrow template")
	feed=[]
	logger = logging.getLogger(__name__)
	img_gray = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY)
	killArrow_matches = detectKillArrow(img_BGR.copy(),killArrow_img.copy())
	for match in killArrow_matches:
		try:
			killers_img, killed_img = cropAroundKillArrow(img_gray, match) 
			killers_img_BGR, killed_img_BGR = cropAroundKillArrow(img_BGR.copy(), match) 
			killData = {
				"killed":"NULL",
				"killed.team":"NULL",
				"killer":"NULL",
				"killer.team":"NULL",
				"assists":"NULL",
				"is.crit":False,
				"ability":"NULL",
				"match_coordinates":str(list(match))
			}
			killed_data = detectHero(killed_img, isLong=None)
			if killed_data is None:
				
				logger.warn(f"No hero found, potential false positive detected.")
			else:
				killed, killed_match, _, killed_cropType = killed_data
				isLong = killed_cropType == "LONG"
				_ , killed_name_img_BGR = cropAroundMatch(killed_img_BGR, killed_match)

				killed_team_color = detectKilledTeamColor(killed_name_img_BGR)

				team_killed = convertColorToTeam(killed_team_color, team_A_color, team_B_color)
				killData["killed"] = str(killed)
				killData["killed.team"] = str(team_killed)

				
				killer_data = detectHero(killers_img, isLong=isLong)

				if killer_data is not None:
					killer, killer_match, _, _ = killer_data
					# this is not a suicide.
					killer_name_img, assistAndAbility_img  = cropAroundMatch(killers_img, killer_match)
					killer_name_img_BGR, _  = cropAroundMatch(killers_img_BGR, killer_match)

					killer_team_color = detectKillerTeamColor(killer_name_img_BGR)


					team_killer = convertColorToTeam(killer_team_color, team_A_color, team_B_color)
					
					killData["killer"] = str(killer)
					killData["killer.team"] = str(team_killer)

				
					assists_names, assists_matches = detectAssists(assistAndAbility_img, isLong=isLong)
					killData["assists"] = ",".join(assists_names)


					if len(assists_names) == 0:
						ability = detectAbility(assistAndAbility_img, killer)
					else:
						rightmost_match = max(assists_matches, key=lambda match: match[2])
						_, abilityImg = cropAroundMatch(assistAndAbility_img, rightmost_match)
						ability = detectAbility(abilityImg, killer)

					killData["ability"] = str(ability)
					
					killData["is.crit"] = not is_kill_arrow_white(img_BGR.copy(), killArrow_img.copy(), match)
				

			logger.debug("Kill detected:")
			logger.debug(f" - Killed: {killData['killed']}")
			logger.debug(f" - Killed team: {killData['killed.team']}")
			logger.debug(f" - Killer: {killData['killer'] }")
			logger.debug(f" - Killer team: {killData['killer.team']}")
			logger.debug(f" - Killer assists: {killData['assists']}")
			logger.debug(f" - Ability used: {killData['ability']}")
			logger.debug(f" - Is crit: {killData['is.crit']}")
			feed.append(killData)
		except Exception as err:
			logger.error(err)
			continue
	return feed

def loadImagesPathsFromDirectory(dir):
	cache = { f.replace(".png",""):join(dir, f) \
		for f in listdir(join(dir)) if isfile(join(dir, f))}
	return cache

def loadIconsSrcCache():
	return loadImagesPathsFromDirectory("assets/icons/")

def loadAssistsSrcCache():
	return loadImagesPathsFromDirectory("assets/assists/")

def loadCharasSrcCache():
	return loadImagesPathsFromDirectory("assets/charas/")

def convertBGRToLabColor(rgbColor):
	rgbColor = sRGBColor(rgbColor[2]/255, rgbColor[1]/255, rgbColor[0]/255)
	return convert_color(rgbColor, LabColor);

def loadAbilitiesSrcCache():
	cache = {}
	abilitiesDir = "assets/abilities/"
	for hero_name in listdir(abilitiesDir):
		if isfile(join(abilitiesDir, hero_name)):
			# we ignore files in the first level
			continue
		for ability_src in listdir(join(abilitiesDir, hero_name+"/")):
			# we are only interested in .png files
			if not ability_src.endswith(".png"): continue
			
			if cache.get(hero_name, None) is None:
				cache[hero_name] = {}
			
			ability_name = ability_src.replace(".png","")
			cache[hero_name][ability_name] = join(abilitiesDir, hero_name+"/", ability_src)
	return cache
	
def writeCSVLine(output, data:List[str]):
	output.write('\n'+",".join(data));


# Those algorithms are used for feature matching, used by the ultimate detection system.
kaze = cv2.KAZE_create(extended=True, upright=True , threshold=1e-6)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(None, search_params)

UltimateFeatureCache = loadUltimateFeatureCache()

# Those file paths caches are used by the killfeed parsing system.
abilities_src = loadAbilitiesSrcCache()
icons_src = loadIconsSrcCache()
assists_src = loadAssistsSrcCache()
kill_arrow_img = cv2.imread('assets/killfeed/killarrow.png',0)

logger_img = logging.getLogger("killfeed")
logger_img.propagate = False
fh = FileHandler('logs/extracting_killfeed.html', mode="w")
logger_img.setLevel(logging.DEBUG)
logger_img.addHandler(fh)

import sqlite3


DB_PATH = "data/processed/MetaWatch_statistics.sqlite3"

# Prepare the database for data dumping


from .BaseExtractor import BaseExtractor as BaseExtractor
from .HealthExtractor import HealthExtractor as HealthExtractor
from .KillfeedExtractor import KillfeedExtractor as KillfeedExtractor
from .NicknameExtractor import NicknameExtractor as NicknameExtractor
from .TeamnameExtractor import TeamnameExtractor as TeamnameExtractor
from .UltimateExtractor import UltimateExtractor as UltimateExtractor

