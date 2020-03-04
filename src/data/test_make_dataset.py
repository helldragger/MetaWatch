from .extraction import readKillFeed, convertImgToTeamname, convertImgToText, convertUltimateImgToValue, convertHeroHealthImgToNotches, convertNotchesToColors, convertBGRToLabColor;

import numpy as np
import cv2
from matplotlib import pyplot as plt
import seaborn as sns

import logging


def test_healthBarParsing():
	logger = logging.getLogger("test_healthBarParsing")
	log_fmt = '[%(asctime)s] [%(name)s] [%(levelname)s] : %(message)s'
	logging.basicConfig(level=logging.INFO, format=log_fmt)
	from logging import FileHandler
	fh1 = FileHandler('logs/tests_extracting_healthbars.log', mode="w")
	logger.addHandler(fh1)
	logger.setLevel(logging.INFO)

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
	countExpected = {}
	for src in notchesTypesExpected.keys():
		countExpected[src] = 0
		for color in notchesTypesExpected[src].keys():
			countExpected[src] += notchesTypesExpected[src][color];
	colorToIndex = {"white":0,"yellow":1, "red":2, "blue":3}
	imgs = {}
	# 1. we 
	countError = 0
	colorError = 0
	
	countTotal = 0
	colorTotal = 0
	
	errors = np.zeros((len(notchesTypesExpected.keys()), 4))
	i = 0
	legendPrinted = False
	for src in notchesTypesExpected.keys():
		logger.info(f":: TEST {src} ::")
		healthBarImg = cv2.imread('assets/tests/Health/'+src+'.jpg',-1);
		#healthBarImg = cv2.cvtColor(healthBarImg, cv2.COLOR_BGR2RGB )
		assert healthBarImg is not None
		labels = convertHeroHealthImgToNotches(healthBarImg)
		count = labels[1]
		colors = convertNotchesToColors(labels, healthBarImg)
		local_count_error = 0
		local_color_error = 0
		expectedColors = notchesTypesExpected[src]
		for colorKey in expectedColors.keys():
			colorTotal += expectedColors[colorKey]
			countTotal += expectedColors[colorKey]
			local_color_error += abs(expectedColors[colorKey] - colors[colorKey])
			colorError += local_count_error
		local_count_error =  abs(countExpected[src] - count)
		local_color_error = max(0, local_color_error-local_count_error)

		countError += local_count_error
		# error detection
		for color in colors.keys():
			errors[i, colorToIndex[color]] += abs(expectedColors[color] - colors[color])  
		i+=1

		if local_count_error > 0 or local_color_error > 0:
			logger.info(f"\t- count absolute error : {local_count_error}")
			logger.info(f"\t- color absolute error : {local_color_error}")
		else:
			logger.info(f"\tO.K.")


	# Acceptance 90%+ accuracy
	countAcc = 1 -(countError / countTotal)
	logger.info(f"Count detection accuracy: {countAcc*100} %")
	assert countAcc >= 0.90 # 10% d'erreur

	sns.heatmap(data=errors, xticklabels=["white", "yellow",  "red", "blue"]).get_figure().savefig("tests/output_healthBarParsing_ErrorHeatmap.png")
	
	logger.info(f"total errors: {errors.sum(axis=0)}")

	colorAcc = 1 - (colorError / colorTotal)
	logger.info(f"Color detection accuracy: {colorAcc*100} %")
	assert colorAcc >= 0.90 # 10% d'erreur

from os import listdir
from os.path import isfile, join

def test_NicknameParsing():  
	logger = logging.getLogger("test_NicknameParsing") 
	log_fmt = '[%(asctime)s] [%(name)s] [%(levelname)s] : %(message)s'
	logging.basicConfig(level=logging.INFO, format=log_fmt)
	from logging import FileHandler
	fh1 = FileHandler('logs/tests_extracting_nickname.log', mode="w")
	logger.addHandler(fh1)
	logger.setLevel(logging.INFO)
	imgDir = "assets/tests/Nickname/"
	
	srcToIndex = {}
	srcIndex = []
	srcToPath = {}
	for f in listdir(imgDir) :
		if isfile(join(imgDir, f)):
			src = join(imgDir, f)
			f = f.replace(".png", "")
			srcToIndex[f] = len(srcIndex)
			srcIndex.append(f)
			srcToPath[f] = src
	errors = np.zeros((len(srcToPath),1))

	for src in srcToPath.keys():
		logger.info(f":: TEST {src} ::")
		img = cv2.cvtColor(cv2.imread(srcToPath[src]), cv2.COLOR_BGR2RGB)
		expected, version = src.split("-")
		result = convertImgToText(img)
		if result != expected:
			logger.error(f"\t-{expected}")
			logger.error(f"\t+{result}")
			errors[srcToIndex[src],0] = 1
		else:
			logger.info("\tO.K.")

	sns.heatmap(data=errors, yticklabels=srcIndex, xticklabels=["Errors"]).get_figure().savefig("tests/output_NicknameParsing_ErrorHeatmap.png")

	totalErrors =  errors.sum(axis=0)[0]
	logger.info(f"Total errors: {totalErrors}")
	accuracy = (1 - (totalErrors / len(srcToPath)))
	logger.info(f"Nickname detection accuracy: {accuracy*100} %")
	assert accuracy >= 0.80 # 20% d'erreur


def test_UltimateParsing():
	logger = logging.getLogger("test_UltimateParsing")
	log_fmt = '[%(asctime)s] [%(name)s] [%(levelname)s] : %(message)s'
	logging.basicConfig(level=logging.INFO, format=log_fmt)
	from logging import FileHandler
	fh1 = FileHandler('logs/tests_extracting_ultimate.log', mode="w")
	logger.addHandler(fh1)
	logger.setLevel(logging.INFO)
	# test format: expectedValue (number).png
	imgDir = "assets/tests/Ultimate/"
	onlyfiles = [f for f in listdir(imgDir) if isfile(join(imgDir, f))]
	
	expected = {}
	testedValues = set();
	valueToIndex = {}
	valueLabels = []
	for src in onlyfiles:
		value  = src.split(" ")[0];
		expected[src] = (value == "100");
		if value not in testedValues:
			valueToIndex[src] = len(testedValues)
			testedValues.add(src)
			valueLabels.append(src)
	errors = np.zeros((len(testedValues),  2))
	for src in expected.keys():
		logger.info(f":: TEST {src} ::")
		img = cv2.imread('assets/tests/Ultimate/'+src,-1);
		value = convertUltimateImgToValue(img)
		if value != expected[src]:
			logger.error(f"\t-{expected[src]}")
			logger.error(f"\t+{value}")
			errors[valueToIndex[src],  0 if value else 1] += 1
		else:
			logger.info("\tO.K.")

	sns.heatmap(data=errors, xticklabels=["Expected: False", "True"], yticklabels=valueLabels).get_figure().savefig("tests/output_UltimateParsing_ErrorHeatmap.png")

	totalErrors =  errors.sum(axis=0)[0]
	logger.info(f"Total errors: {totalErrors}")
	accuracy = (1 - (totalErrors / len(expected)))
	logger.info(f"Ultimate detection accuracy: {accuracy*100} %")
	assert accuracy >= 0.95 # 15% d'erreur


def test_teamnameParsing():
	logger = logging.getLogger("test_teamnameParsing")
	log_fmt = '[%(asctime)s] [%(name)s] [%(levelname)s] : %(message)s'
	logging.basicConfig(level=logging.INFO, format=log_fmt)
	from logging import FileHandler
	fh1 = FileHandler('logs/tests_extracting_teamname.log', mode="w")
	logger.addHandler(fh1)
	logger.setLevel(logging.INFO)
	imgDir = "assets/tests/Team/"
	imgs = {}
	srcToIndex = {}
	srcIndex = []
	for f in listdir(imgDir) :
		if isfile(join(imgDir, f)):
			src = join(imgDir, f)
			imgs[f] = cv2.cvtColor(cv2.imread(src), cv2.COLOR_BGR2RGB)
			srcToIndex[f] = len(srcIndex)
			srcIndex.append(f)
	errors = np.zeros((len(imgs),1))

	for src in imgs.keys():
		logger.info(f":: TEST {src} ::")
		expected, version = src.split(" ")
		expected = expected.replace("_", "")
		result = convertImgToTeamname(imgs[src])
		if result != expected:
			logger.error(f"\t-{expected}")
			logger.error(f"\t+{result}")
			errors[srcToIndex[src],0] = 1
		else:
			logger.info("\tO.K.")

	
	sns.heatmap(data=errors, yticklabels=srcIndex, xticklabels=["Errors"]).get_figure().savefig("tests/output_teamnameParsing_ErrorHeatmap.png")
	
	totalErrors =  errors.sum(axis=0)[0]
	logger.info(f"Total errors: {totalErrors}")
	accuracy = (1 - (totalErrors / len(imgs)))
	logger.info(f"Team name detection accuracy: {accuracy*100} %")
	assert accuracy >= 0.85 # 




def test_readKillFeed():
	logger = logging.getLogger("test_readKillFeed")
	log_fmt = '[%(asctime)s] [%(name)s] [%(levelname)s] : %(message)s'
	logging.basicConfig(level=logging.INFO, format=log_fmt)
	from logging import FileHandler
	fh1 = FileHandler('logs/tests_extracting_killfeed.log', mode="w")
	logger.addHandler(fh1)

	logger_img = logging.getLogger("killfeed")
	logger_img.propagate = False
	fh2 = FileHandler('logs/tests_extracting_killfeed.html', mode="w")
	logger_img.addHandler(fh2)

	logger_img.setLevel(logging.INFO)
	logger.setLevel(logging.INFO)

	killfeedExpected = {
		"1":{
			"feed":[
				{
					"killed":"ana",
					"killed.team":"A",
					"killer":"hammond",
					"killer.team":"B",
					"assists":"ana,mercy",
					"is.crit":False,
					"ability":"1",
				},
				{
					"killed":"sombra",
					"killed.team":"B",
					"killer":"dva",
					"killer.team":"A",
					"assists":"",
					"is.crit":False,
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
					"is.crit":False,
					"ability":"none",
				},
				{
					"killed":"mercy",
					"killed.team":"B",
					"killer":"dva",
					"killer.team":"A",
					"assists":"",
					"is.crit":False,
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
					"is.crit":False,
					"ability":"3",
				},
				{
					"killed":"widowmaker",
					"killed.team":"B",
					"killer":"reaper",
					"killer.team":"A",
					"assists":"",
					"is.crit":True,
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
					"is.crit":True,
					"ability":"none",
				},
				{
					"killed":"ana",
					"killed.team":"B",
					"killer":"mercy",
					"killer.team":"B",
					"assists":"",
					"is.crit":False,
					"ability":"2",
				}
			],
			"team_A_color":np.array([233,233,233]),
			"team_B_color":np.array([31,154,233])
		},
		"5":{
			"feed":[
				{
					"killed":"pharah",
					"killed.team":"A",
					"killer":"winston",
					"killer.team":"B",
					"assists":"",
					"is.crit":False,
					"ability":"1",
				}, # IS HIDDEN BY A LIGHTNING ON THE ARROW
				{
					"killed":"lucio",
					"killed.team":"B",
					"killer":"sombra",
					"killer.team":"A",
					"assists":"",
					"is.crit":False,
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
					"is.crit":False,
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
					"is.crit":False,
					"ability":"3",
				},
				{
					"killed":"ana",
					"killed.team":"B",
					"killer":"pharah",
					"killer.team":"A",
					"assists":"sombra",
					"is.crit":False,
					"ability":"3",
				} # IS HIDDEN BY A LIGHTNING ON THE ARROW
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
					"is.crit":True,
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
					"is.crit":True,
					"ability":"none",
				},
				{
					"killed":"orisa",
					"killed.team":"A",
					"killer":"mercy",
					"killer.team":"A",
					"assists":"",
					"is.crit":False,
					"ability":"2",
				},
				{
					"killed":"orisa",
					"killed.team":"A",
					"killer":"widowmaker",
					"killer.team":"B",
					"assists":"orisa",
					"is.crit":True,
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
					"is.crit":False,
					"ability":"none",
				},
				{
					"killed":"meka",
					"killed.team":"B",
					"killer":"zenyatta",
					"killer.team":"A",
					"assists":"",
					"is.crit":False,
					"ability":"none",
				}
			],
			"team_A_color":np.array([233,233,233]),
			"team_B_color":np.array([205,104,68])
		},
	}
	imgDir = "assets/tests/Killfeed/"
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

		
		logger.info(f":: TEST {killfeed_name} ::")
		expected_data = killfeedExpected[killfeed_name]
		killfeed_img = cv2.imread(join(imgDir, killfeed_name+'.png'))
		rawResult = readKillFeed(killfeed_img, killArrow_img.copy(), convertBGRToLabColor(expected_data["team_A_color"]), convertBGRToLabColor(expected_data["team_B_color"]))
		
		expected = expected_data["feed"];
		logger.debug(f"expected: {expected}")
		result = list(filter(lambda data: data["killed"]!="NA", rawResult))
		logger.info(f"{len(rawResult) -len (result)} false positives detected.")
		logger.debug(f"result after removing false positives: {result}")
		len_diff = len(expected) - len(result)

		if len_diff != 0:
			logger.error(f"{abs(len_diff)} {'LESS' if len_diff < 0 else 'MORE'} feed element expected")
			# a missing or a duplicata in the feed counts as errors for each properties 
			for property_name in properties:
				errors[expectedToIndex[killfeed_name],propertyToIndex[property_name]] += abs(len_diff) 
		for i in range(min(len(expected), len(result))):
			logger.info(f".subtest {i}")
			result_kill = result[i]
			expected_kill = expected[i]
			errored = False
			for property_name in expected_kill.keys():
				expected_value = expected_kill[property_name]
				result_value = result_kill.get(property_name, None)
				if expected_value != result_value:
					errored = True
					logger.error(f"\t{property_name}: \n\t\t-{expected_value}\n\t\t+{result_value}")
					errors[expectedToIndex[killfeed_name],propertyToIndex[property_name]]  += 1
			if not errored:
				logger.info(f"\tO.K.")
	sns.heatmap(data=errors, yticklabels=expectedIndex, xticklabels=propertyIndex).get_figure().savefig("tests/output_killfeedParsing_ErrorHeatmap.png")

	totalErrors =  errors.sum()
	logger.info(f"Total errors: {totalErrors}")
	accuracy = (1 - (totalErrors / maxErrors))
	logger.info(f"Kill feed detection accuracy: {accuracy*100} %")
	assert accuracy > 0.7 #