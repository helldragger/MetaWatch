#%%
#from src.data.replayAnalyzer import readKillFeed;
import cv2
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import isfile, join
import math
from matplotlib import pyplot as plt
import seaborn as sns

from logging import FileHandler
from vlogging import VisualRecord

import logging
#logger = logging.getLogger("demo")
#fh = FileHandler('notebooks/logs/killfeed.html', mode="w")
#logger.setLevel(logging.DEBUG)
#logger.addHandler(fh)

import pytesseract
#%%
image = cv2.imread("data/raw/3/truth.PNG")

# SALIENCY TEST
# initialize OpenCV's static fine grained saliency detector and
# compute the saliency map
saliency = cv2.saliency.StaticSaliencyFineGrained_create()
(success, saliencyMap) = saliency.computeSaliency(image)
 
# if we would like a *binary* map that we could process for contours,
# compute convex hull's, extract bounding boxes, etc., we can
# additionally threshold the saliency map
threshMap = cv2.threshold(saliencyMap.astype("uint8"), 128, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
 
# show the images
cv2.imshow("Image", image)
cv2.imshow("Output", saliencyMap)
cv2.imshow("Thresh", threshMap)
cv2.waitKey(0)



#%% 
# TEMPLATE MATCHING TEST
image = cv2.imread("data/raw/3/truth.PNG", 0)
img2 = image.copy()
template = cv2.imread('assets/killfeed/killarrow.png',0)
w, h = template.shape[::-1]
# All the 6 methods for comparison in a list
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
			'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
for meth in methods:
	img = img2.copy()
	method = eval(meth)
	# Apply template Matching
	res = cv2.matchTemplate(img,template,method)
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
	# If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
	if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
		top_left = min_loc
	else:
		top_left = max_loc
	bottom_right = (top_left[0] + w, top_left[1] + h)
	cv2.rectangle(img,top_left, bottom_right, 255, 2)
	plt.subplot(121),plt.imshow(res,cmap = 'gray')
	plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.imshow(img,cmap = 'gray')
	plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
	plt.suptitle(meth)
	plt.show()
	print("coordinates, top_left:", str(top_left), "; bottom_right:", str(bottom_right))
	expected = {"top_left":(1728, 222), "bottom_right":(1746, 248)}
#%%

# detection of heroes. loading sources.


imgDir = "assets/"
assistsDir = imgDir+"assists/"
charasDir = imgDir+"charas/"
iconsDir = imgDir+"icons/"
abilitiesDir = imgDir+"abilities/"

assists_src = { f.replace(".png",""):join(assistsDir, f) \
	for f in listdir(join(assistsDir)) if isfile(join(assistsDir, f))}

charas_src = { f.replace(".png",""):join(charasDir, f) \
	for f in listdir(join(charasDir)) if isfile(join(charasDir, f))}
	
icons_src = { f.replace(".png",""):join(iconsDir, f) \
	for f in listdir(join(iconsDir)) if isfile(join(iconsDir, f))}

abilities_src = {}

for hero_name in charas_src.keys():
	try:

		for ability_src in listdir(join(abilitiesDir, hero_name+"/")):
			if abilities_src.get(hero_name, None) is None:
				abilities_src[hero_name] = {}
			ability_name = ability_src.replace(".png","")
			print(hero_name," > DETECTED ABILITY",ability_name)
			abilities_src[hero_name][ability_name] = join(abilitiesDir, hero_name+"/", ability_src)
	except FileNotFoundError as err:
		print(hero_name," > NO ABILITIES REGISTERED")


#%%
MATCH_COLOR = (100,100,255)
MATCH_WEIGHT = 1

def logMatchesOnBGRImg(img_BGR, matches, title="Matches", text="", step=""):
	img = img_BGR.copy()
	for match in matches:
		cv2.rectangle(img,(match[0],match[1]),(match[2],match[3]),MATCH_COLOR,MATCH_WEIGHT)
	logger.debug(VisualRecord(title+step, img, text, fmt="png"))
	
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
	
	logger.debug(VisualRecord(title, [template, result_img], text, fmt="png"))

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
	
	logger.debug(VisualRecord(title, [template, result_img], text, fmt="png"))

def logCroppedTemplateMatchOnGrayImg(img_Gray, template, loc, cropType, title="scaled template match (template, match)", text=""):
	img_BGR = cv2.cvtColor(img_Gray, cv2.COLOR_GRAY2BGR)
	logCroppedTemplateMatchOnBGRImg(img_BGR, template, loc, cropType, title, text)

def logCroppedTemplateMatch(img, template, loc, cropType, title="scaled template match (template, match)", text=""):
	if len(img.shape) == 2:
		logCroppedTemplateMatchOnGrayImg(img, template, loc, cropType, title, text)
	elif len(img.shape) == 3:
		logCroppedTemplateMatchOnGrayImg(img, template, loc, cropType, title, text)

def logImg(img, title="", text=""):
	logger.debug(VisualRecord(title,img,text, fmt="png"))

################################LOGGING

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
	
	#i = 0
	first = True
	for match in matches:
		#print("PROGRESS> processing match",i)
		#print(match)
		#i+=1
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
	res = cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
	#res[res < max(threshold, np.max(res)*tolerance)] = 0
	if show:
		#cv2.imshow("template", template)
		cv2.imshow("img", img)
		print("min",np.min(res),"max", np.max(res),"mean", np.mean(res))
		cv2.imshow("res",res)
		res_threshold = res
		res_tolerance = res

		res_threshold[res_threshold < threshold] = 0
		res_tolerance[res_tolerance < np.max(res_tolerance)*tolerance] = 0
		
		cv2.imshow("res, threshold",res_threshold)
		cv2.imshow("res, tolerance",res_tolerance)

		print(len(np.where (res >= threshold)))
		print(len(np.where (res >= np.max(res)*tolerance)))
		cv2.waitKey(0)

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
	normal_loc = None
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
	result_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	
	
	cv2.rectangle(result_img,(maxLoc[0],maxLoc[1]),(maxLoc[0]+w,maxLoc[1]+h),(100,100,255),3)
	text =  "val = " +str(round(_maxVal, 3))
	
	logger.debug(VisualRecord("Best template match (template, match)", [template, result_img], text, fmt="png"))


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
					logCroppedTemplateMatch(cropped_img, hero_img, match, cropType, title="HERO UPDATED:"+hero_name, text="value @"+str(value))
					best_hero = (hero_name, match, value, cropType)
					best_value = value
				else:
					logCroppedTemplateMatch(cropped_img, hero_img, match, cropType, title="HERO IGNORED: "+hero_name, text="value @"+str(value))
						
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
					logCroppedTemplateMatch(cropped_img, hero_img, match, cropType, title="ASSIST DETECTED: "+hero_name, text="value @"+str(value))
					assists.append(hero_name)
					assists_matches.append(match)
				else:
					logCroppedTemplateMatch(cropped_img, hero_img, match, cropType, title="ASSIST IGNORED: "+hero_name, text="value @"+str(value))

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

	
def detectKilledTeamColor(killed_name_img_BGR):
	maxY, _, _ = killed_name_img_BGR.shape
	# we center on the vertical axis to be in th emiddle of the kill feed ticket
	# we then offset just a little after the hero split to get the color between the hero frame and the player nickname 
	X = 5
	
	return killed_name_img_BGR[int(maxY*0.5),X,:]

def detectKillerTeamColor(killer_name_img_BGR):
	maxY, _, _ = killer_name_img_BGR.shape
	# we center on the vertical axis to be in th emiddle of the kill feed ticket
	# we then offset just a little after the hero split to get the color between the hero frame and the player nickname 
	X = 5
	return killer_name_img_BGR[int(maxY*0.5),-X,:]

def convertColorToTeam(unknown_color, team_A_color, team_B_color):
	dist_A = np.sum( (unknown_color - team_A_color)**2 )
	dist_B = np.sum( (unknown_color - team_B_color)**2 )
	if dist_A < dist_B:
		return "A"
	else:
		return "B"


def detectKillArrow(img_BGR,killArrow_img):
	img_gray = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2LAB)[:,:,0]
	killArrow_canny = cv2.Canny(killArrow_img,100,200)

	#we remove the leftmost edge, to allow the detection of arrows combined to other abilities images
	killArrow_canny_noedge = killArrow_canny[3:,:]
	
	img_canny = cv2.Canny(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2LAB)[:,:,0],100,200)
	
	edge_matches = getTemplateMatches(img_canny, killArrow_canny, show=False, threshold=0.4)
	logMatchesOnGrayImg(img_canny, edge_matches, "edge only kill arrow matches")
	#print(edge_matches)

	noedge_matches = getTemplateMatches(img_canny, killArrow_canny_noedge, show=False, threshold=0.4)
	logMatchesOnGrayImg(img_canny, noedge_matches, "no edge only kill arrow matches")
	
	full_matches = getTemplateMatches(img_gray, killArrow_img, show=False, threshold=0.8)
	logMatchesOnBGRImg(img_BGR, full_matches, "full kill arrow matches")
	#print(noedge_matches)
	all_matches = edge_matches
	all_matches.extend(noedge_matches)
	all_matches.extend(full_matches)
	#print(all_matches)
	unique_matches = uniqueMatches(all_matches)
	#print(unique_matches)

	final_matches = sorted(unique_matches, key=lambda loc_data:loc_data[1])
	#print(final_matches)
	logMatchesOnBGRImg(img_BGR, final_matches, "final kill arrow matches")
	return final_matches

def readKillFeed(img_BGR, killArrow_img, team_A_color, team_B_color):
	feed=[]
	img_gray = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY)
	killArrow_matches = detectKillArrow(img_BGR,killArrow_img)
	for match in killArrow_matches:
		killers_img, killed_img = cropAroundKillArrow(img_gray, match) 
		killers_img_BGR, killed_img_BGR = cropAroundKillArrow(img_BGR, match) 
		killData = {
			"killed":"",
			"killed.team":"",
			"killer":"",
			"killer.team":"",
			"assists":"",
			"is.crit":"",
			"ability":"",
		}
		killed_data = detectHero(killed_img, isLong=None)
		if killed_data is None:
			#print("no hero found, arrow false positive")
			#match_img = img_canny
			#match_img = cv2.cvtColor(match_img, cv2.COLOR_GRAY2BGR)
			#cv2.rectangle(match_img,(match[0],match[1]),(match[2],match[3]),(100,100,255),3)
			#cv2.imshow("matched false positive", match_img)
			#cv2.waitKey(0)
			continue; # this is a false positive.
		#print(".......................................")
		killed, killed_match, _, killed_cropType = killed_data
		isLong = killed_cropType == "LONG"
		_ , killed_name_img_BGR = cropAroundMatch(killed_img_BGR, killed_match)

		killed_team_color = detectKilledTeamColor(killed_name_img_BGR)

		team_killed = convertColorToTeam(killed_team_color, team_A_color, team_B_color)
		killData["killed"] = str(killed)
		killData["killed.team"] = str(team_killed)
		
		#print("KILLED >", str(killed), " team",team_killed)


		killer_data = detectHero(killers_img, isLong=isLong)

		if killer_data is not None:
			killer, killer_match, _, _ = killer_data
			# this is not a suicide.
			#cv2.imshow("killers_img", killers_img)
			killer_name_img, assistAndAbility_img  = cropAroundMatch(killers_img, killer_match)
			killer_name_img_BGR, _  = cropAroundMatch(killers_img_BGR, killer_match)

			killer_team_color = detectKillerTeamColor(killer_name_img_BGR)


			team_killer = convertColorToTeam(killer_team_color, team_A_color, team_B_color)
			
			killData["killer"] = str(killer)
			killData["killer.team"] = str(team_killer)
		
			#print("KILLER >", str(killer), " team",team_killer)


			assists_names, assists_matches = detectAssists(assistAndAbility_img, isLong=isLong)
			killData["assists"] = ",".join(assists_names)

			#print("ASSIST >", str(assists_names))

			if len(assists_names) == 0:
				ability = detectAbility(assistAndAbility_img, killer)
			else:
				rightmost_match = max(assists_matches, key=lambda match: match[2])
				_, abilityImg = cropAroundMatch(assistAndAbility_img, rightmost_match)
				ability = detectAbility(abilityImg, killer)

			killData["ability"] = str(ability)
			#print("ABILITY>", str(ability))
			
			is_crit = not is_kill_arrow_white(img_BGR, killArrow_img, match)
			
			killData["is.crit"] = str(is_crit)
			#print("IS CRIT>", str(is_crit))
		#else:
			#print("KILLER > none")
		feed.append(killData)
	return feed
			

#result = readKillFeed(img_BGR, killArrow_img, team_A_color, team_B_color)

def test_readKillFeed():
	killfeedExpected = {
		"1":{
			"feed":[
				{
					"killed":"ana",
					"killed.team":"A",
					"killer":"hammond",
					"killer.team":"B",
					"assists":"ana,mercy",
					"is.crit":"False",
					"ability":"1",
				},
				{
					"killed":"sombra",
					"killed.team":"B",
					"killer":"dva",
					"killer.team":"A",
					"assists":"",
					"is.crit":"False",
					"ability":"none",
				}
			],
			"team_A_color":np.array([233,233,233]),
			"team_B_color":np.array([31,154,233])
		},
		"2":{
			"feed":[
				{
					"killed":"meka",
					"killed.team":"A",
					"killer":"dva",
					"killer.team":"B",
					"assists":"ana",
					"is.crit":"False",
					"ability":"none",
				},
				{
					"killed":"mercy",
					"killed.team":"B",
					"killer":"dva",
					"killer.team":"A",
					"assists":"",
					"is.crit":"False",
					"ability":"none",
				}
			],
			"team_A_color":np.array([233,233,233]),
			"team_B_color":np.array([31,154,233])
		},
		"3":{
			"feed":[
				{
					"killed":"ana",
					"killed.team":"B",
					"killer":"reaper",
					"killer.team":"A",
					"assists":"",
					"is.crit":"False",
					"ability":"3",
				},
				{
					"killed":"widowmaker",
					"killed.team":"B",
					"killer":"reaper",
					"killer.team":"A",
					"assists":"",
					"is.crit":"True",
					"ability":"none",
				}
			],
			"team_A_color":np.array([233,233,233]),
			"team_B_color":np.array([31,154,233])
		},
		"4":{
			"feed":[
				{
					"killed":"mercy",
					"killed.team":"B",
					"killer":"widowmaker",
					"killer.team":"A",
					"assists":"ana",
					"is.crit":"True",
					"ability":"none",
				},
				{
					"killed":"ana",
					"killed.team":"B",
					"killer":"mercy",
					"killer.team":"B",
					"assists":"",
					"is.crit":"False",
					"ability":"2",
				}
			],
			"team_A_color":np.array([233,233,233]),
			"team_B_color":np.array([31,154,233])
		},
		"5":{
			"feed":[
				#{
				#	"killed":"pharah",
				#	"killed.team":"A",
				#	"killer":"winston",
				#	"killer.team":"B",
				#	"assists":"",
				#	"is.crit":"False",
				#	"ability":"1",
				#}, # IS HIDDEN BY A LIGHTNING ON THE ARROW
				{
					"killed":"lucio",
					"killed.team":"B",
					"killer":"sombra",
					"killer.team":"A",
					"assists":"",
					"is.crit":"False",
					"ability":"none",
				}
			],
			"team_A_color":np.array([233,233,233]),
			"team_B_color":np.array([205,104,68])
		},
		"6":{
			"feed":[
				{
					"killed":"pharah",
					"killed.team":"A",
					"killer":"mercy",
					"killer.team":"A",
					"assists":"",
					"is.crit":"False",
					"ability":"2",
				}
			],
			"team_A_color":np.array([233,233,233]),
			"team_B_color":np.array([205,104,68])
		},
		"7":{
			"feed":[
				{
					"killed":"lucio",
					"killed.team":"B",
					"killer":"pharah",
					"killer.team":"A",
					"assists":"sombra",
					"is.crit":"False",
					"ability":"3",
				}#,
				#{
				#	"killed":"ana",
				#	"killed.team":"B",
				#	"killer":"pharah",
				#	"killer.team":"A",
				#	"assists":"sombra",
				#	"is.crit":"False",
				#	"ability":"3",
				#} # IS HIDDEN BY A LIGHTNING ON THE ARROW
			],
			"team_A_color":np.array([233,233,233]),
			"team_B_color":np.array([205,104,68])
		},
		"8":{
			"feed":[
				{
					"killed":"orisa",
					"killed.team":"A",
					"killer":"widowmaker",
					"killer.team":"B",
					"assists":"orisa",
					"is.crit":"True",
					"ability":"none",
				}
			],
			"team_A_color":np.array([31,139,134]),
			"team_B_color":np.array([233,233,233])
		}, 
		"9":{
			"feed":[
				{
					"killed":"hanzo",
					"killed.team":"B",
					"killer":"widowmaker",
					"killer.team":"A",
					"assists":"",
					"is.crit":"True",
					"ability":"none",
				},
				{
					"killed":"orisa",
					"killed.team":"A",
					"killer":"mercy",
					"killer.team":"A",
					"assists":"",
					"is.crit":"False",
					"ability":"2",
				},
				{
					"killed":"orisa",
					"killed.team":"A",
					"killer":"widowmaker",
					"killer.team":"B",
					"assists":"orisa",
					"is.crit":"True",
					"ability":"none",
				}
			],
			"team_A_color":np.array([31,139,134]),
			"team_B_color":np.array([233,233,233])
		}, 
		"10":{
			"feed":[
				{
					"killed":"dva",
					"killed.team":"B",
					"killer":"zarya",
					"killer.team":"A",
					"assists":"",
					"is.crit":"False",
					"ability":"none",
				},
				{
					"killed":"meka",
					"killed.team":"B",
					"killer":"zenyatta",
					"killer.team":"A",
					"assists":"",
					"is.crit":"False",
					"ability":"none",
				}
			],
			"team_A_color":np.array([233,233,233]),
			"team_B_color":np.array([205,104,68])
		},
	}
	imgDir = "assets/tests/killfeed/"
	expectedToIndex = {}
	expectedIndex = []
	propertyToIndex = {}
	propertyIndex = []
	properties = [
		"killed",
		"killed.team",
		"killer",
		"killer.team",
		"assists",
		"is.crit",
		"ability"
	]
	maxErrors = 0
	for killfeed_name in killfeedExpected.keys():
		expectedToIndex[killfeed_name] = len(expectedIndex)
		expectedIndex.append(killfeed_name)
		maxErrors += len(properties) * len(killfeedExpected[killfeed_name]["feed"])
	
	

	for property_name in properties:
		
		propertyToIndex[property_name] = len(propertyIndex)
		propertyIndex.append(property_name)
	
	errors = np.zeros((len(expectedIndex),len(propertyIndex)))
	
	killArrow_img = cv2.imread('assets/killfeed/killarrow.png',0)
	for killfeed_name in killfeedExpected.keys():

		if killfeed_name in ["7"]:
			logger.setLevel(logging.DEBUG)
		else:
			logger.setLevel(logging.INFO)

		print(":: TEST", killfeed_name,"::")
		expected_data = killfeedExpected[killfeed_name]
		killfeed_img = cv2.imread(join(imgDir, killfeed_name+'.png'))
		
		result = readKillFeed(killfeed_img, killArrow_img.copy(), expected_data["team_A_color"], expected_data["team_B_color"])
		expected = expected_data["feed"];
		len_diff = len(expected) - len(result)

		if len_diff != 0:
			print(abs(len_diff),"LESS" if len_diff < 0 else "MORE", "feed element expected")
			# a missing or a duplicata in the feed counts as errors for each properties 
			for property_name in properties:
				errors[expectedToIndex[killfeed_name],propertyToIndex[property_name]] += abs(len_diff) 
		for i in range(min(len(expected), len(result))):
			print(".subtest",i)
			result_kill = result[i]
			expected_kill = expected[i]
			errored = False
			for property_name in expected_kill.keys():
				expected_value = expected_kill[property_name]
				result_value = result_kill.get(property_name, None)
				if expected_value != result_value:
					errored = True
					print("\t",property_name, ": \n\t\t-",expected_value," \n\t\t+",result_value)
					errors[expectedToIndex[killfeed_name],propertyToIndex[property_name]]  += 1
			if not errored:
				print("\tO.K.")
			print("")
			#printDiff(result_kill, expected_kill)
	sns.heatmap(data=errors, yticklabels=expectedIndex, xticklabels=propertyIndex).get_figure().savefig("tests/output_killfeedParsing_ErrorHeatmap.png")

	totalErrors =  errors.sum()
	print("Total errors: ", totalErrors)
	accuracy = (1 - (totalErrors / maxErrors))
	print("Accuracy: ", accuracy,"%")
	assert accuracy >= 0.95 # 


# cropping of image around each kill arrow match
img_gray = cv2.imread("data/raw/3/truth.PNG", 0)
img_BGR = cv2.imread("data/raw/3/truth.PNG", -1)
killArrow_img = cv2.imread('assets/killfeed/killarrow.png',0)

team_A_color = np.array([255,237,217])
team_B_color = np.array([53 ,171 ,234])

logger = logging.getLogger("demo")

log_file = FileHandler('notebooks/logs/killfeed.html', mode="w")
logger.setLevel(logging.INFO)
logger.addHandler(log_file)
test_readKillFeed()
logging.shutdown()