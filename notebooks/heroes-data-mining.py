#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'notebooks'))
	print(os.getcwd())
except:
	pass

#%%
# Objectif 1: recuperer les données basiques sur les héros. (armure, vie, etc)
# Objectif 2: recuperer les données sur les leurs compétences (type, action, cibles, dégâts, effets....).
from typing import Dict, Union, List, Tuple


#%%
import requests

# testing overwatch-api.net API

heroesListing = requests.get("http://overwatch-api.net/api/v1/hero").json()
print(heroesListing)

#%%
heroesData = {}
for i in range(1, heroesListing["total"]+1):
    heroData = requests.get("http://overwatch-api.net/api/v1/hero/"+str(i)).json()
    heroName = heroData["name"]
    heroesData[heroName] = heroData

print(heroesData)
#%%
from bs4 import BeautifulSoup as BS
from bs4 import Tag

# testing https://overwatch.fandom.com/wiki/Nano_Boost requests
abilitiesExamples = ['Storm_Bow']#, 'Scatter_Arrow', 'Sonic_Arrow', 'Wall_Climb', 'Dragonstrike']
for abilityName in abilitiesExamples: 
    abilityData = requests.get("https://overwatch.fandom.com/wiki/"+abilityName)
    soup = BS(abilityData.content, "html.parser")
    abilityDataDiv = soup.find_all("div", {"class": "tooltip-content"})[0]
    print(abilityDataDiv.prettify())
#%%

heroAbilitiesNames = []

for heroData in heroesData.values():
    heroAbilities = heroData["abilities"]
    heroAbilitiesNames.extend(map(lambda ability : ability["name"].replace(" ", "_"), heroAbilities))
#%%
abilitiesNamesToDataDivs: Dict[str, Tag] = {} # temporary storage
i = 0
for abilityName in heroAbilitiesNames:    
    i+=1
    print("["+str(i)+"/"+str(len(heroAbilitiesNames))+"]","Fetching", abilityName, "from the wiki...")
    abilitiesNamesToDataDivs[abilityName] = BS(requests.get("https://overwatch.fandom.com/wiki/"+abilityName).content, "html.parser").find_all("div", {"class": "tooltip-content"})[0]

print("All abilities have been fetched!")

#%%
# Parsing abilities data.
# tesing with a single datadiv
abilityName = heroAbilitiesNames[0]
abilityDataDiv:Tag = abilitiesNamesToDataDivs[abilityName]

import re

def extractAbilityData(abilityName:str, abilityDiv:Tag, showResults:bool=True, showErrors:bool=True)->Dict[str, Union[bool, str, Tag, int]]:
    #can headshot?
    totalData = 0
    headshotRegex = re.compile("(?P<possibility>Can|Cannot) headshot")
    # format (1.23|12) - (1.23|12) measure 
    intervalRegex = re.compile("(?P<min>[0-9]+(.[0-9]+|)) - (?P<max>[0-9]+(.[0-9]+|)) (?P<measure>.*)")
    def intervalManager(data, result):
        data[result.group("measure")+" min"]= result.group("min")
        data[result.group("measure")+" max"]= result.group("max")
        return data

    # format (1.23|12%|∞)( |-)measure
    numericValue = re.compile("(?P<value>(([0-9]+(.[0-9]+|))|∞)(%*))( |-)(?P<measure>.*)")
    def numericalManager(data, result):
        data[result.group("measure")]= result.group("value")
        return data


    characteristicTypeRegex = re.compile("(?P<attribute>.* )(?P<type>[a-zA-Z]+) type")
    def characteristicManager(data, result):
        data[result.group("type")+" type"] = result.group("attribute")
        return data

    typeRegex = re.compile("(?P<attribute>[a-zA-Z]+) type")
    def typeManager(data, result):
        data["type"] = result.group("attribute")
        return data

    projectileTypeRegex = re.compile("(?P<attribute>.+) projectile")
    def projectileManager(data, result):
        data["projectile type"] = result.group("attribute")
        return data

    durationRegex = re.compile("Lasts (?P<condition>.*)")
    def durationManager(data, result):
        data["Lasts"] = result.group("condition")
        return data
    
    immobilizationRegex = re.compile("Caster is immobilized")
    def immobilizationManager(data, result):
        data["Caster is immobilized"]  = "true"
        return data

    
    burstRegex = re.compile("All remaining rounds per burst")
    def burstManager(data, result):
        data["All remaining rounds per burst"]  = "true"
        return data
    # Num1, Num2, ..., or NumN measure2 || measure1 Num1, Num2, ...., or NumN text
    #multipleNumeralsRegex = re.compile("(?P<measure1>([a-zA-Z]+ |))(?P<dataTuple>(([0-9]+, )+)or [0-9]+)(?P<measure2>.+)")
    #def multipleNumeralsManager(data, result):
    #    if result.group("measure1") == "Lasts ":
    #        data["Lasts"] = dataTuple
    #    else
    #    
    #    return data

    punchlineRegex = re.compile("\".*|.*\"")
    
    regexTests = [
        (
            intervalRegex,
            intervalManager
        ),
        (
            characteristicTypeRegex,
            characteristicManager
        ),
        (
            projectileTypeRegex,
            projectileManager
        ),
        (
            typeRegex,
            typeManager
        ),
        (
            durationRegex,
            durationManager
        ),
        (
            numericValue,
            numericalManager
        ),
        (
            immobilizationRegex,
            immobilizationManager
        ),
        (
            burstRegex,
            burstManager
        )

    ]

    erroredData = []
    data = {"name" : abilityName}

    regexResult = headshotRegex.search(abilityDiv.get_text())
    if regexResult:
        data["headshot"] = regexResult.group("possibility")        
        totalData += 1


    for characteristic in abilityDataDiv.find_all("i"):
        totalData += 1
        charLine: str = characteristic.get_text()
        # some measures have multiple related data, ex Hack from Sombra can last 6, 20 or 60 sec depending on the situation
        # those measures are rewritten as "data1|data2|data3". instead of "data1, data2, or data3"
        # Same thing concerning multiple tags, "tag1 and tag2" becomes "tag1|tag2"
        charLine = charLine.replace(", or ", "|").replace(", and ", "|").replace(" and ", "|").replace(", ", "|").replace(" or ", "|")

        # some measures have a simple comma between data, we separate them and check them individually.
        for charText in charLine.split(","):
            # if punchline, ignore.
            if punchlineRegex.search(charText):
                continue
            error = True
            for testRegex, testHandler in regexTests:
                regexResult = testRegex.search(charText)
                if regexResult:
                    data = testHandler(data, regexResult)
                    error = False
                    break
            if error:
                if showErrors:
                    print("[ERR] @", abilityName, "-> Cannot parse \""+charText+"\"")
                erroredData.append(charText)
    if showResults:           
        print("["+str(totalData-len(erroredData))+"/"+str(totalData)+"]", (totalData-len(erroredData))/totalData*100, "% Correctly parsed data")
    return data, erroredData

resultsData, resultsErroredData = extractAbilityData(abilityName, abilityDataDiv)

#%%

#extract all abilities Data.

abilitiesData = []
erroredData = {}
errorCount = 0
dataCount = 0
test= {}

for abilityName, abilityDataDiv in abilitiesNamesToDataDivs.items():
    resultsData, resultsErroredData = extractAbilityData(abilityName, abilityDataDiv, showResults=False)
    abilitiesData.append(resultsData)
    erroredData[abilityName] = resultsErroredData
    errorCount += len(resultsErroredData)
    dataCount += len(resultsData.values()) 

print("\nData extracted.\n", errorCount,"errors;", dataCount, "data parsed;", dataCount/(dataCount+errorCount)*100,"% coverage")
import json
with open("../data/processed/abilitiesData.json", 'w') as savefile:
    json.dump(abilitiesData, savefile)
    
#%%
# Objective, clean up and normalize the data set.