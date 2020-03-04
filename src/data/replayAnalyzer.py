
import cv2

from contextlib import ExitStack

from extraction import loadVideo, dateToFrame,frameToDate, loadMaskCache, HealthExtractor, NicknameExtractor, KillfeedExtractor, UltimateExtractor, TeamnameExtractor
import concurrent.futures
import multiprocessing
from math import floor

def runExtractor(extractor, lock):
	extractor.setup(lock)
	print(f"{extractor.name} started")
	extractor.run()
	print(f"{extractor.name} finished")
	
def convertingVideoToImages2(input_filepath:str, output_filepath:str, profiling:bool, health:bool, ultimate:bool, nickname:bool, heroes:bool, teamname:bool, killfeed:bool, exportHealth:bool, exportUltimate:bool, exportNickname:bool, exportTeam:bool, exportHeroes:bool, exportKillfeed:bool, startTime:str, endTime:str, vid_UUID:int, analysis_UUID:int):
	#   input format:
	# input_filepath/
	#   > replay.mp4
	#   > masks/
	#       > Health/
	#           > A1.png
	#           > A2.png
	#           > A3.png
	#           > ...
	#       > Nickname/
	#           > A1.png
	#           > A2.png
	#           > A3.png
	#           > ...
	#       > Ultimate/
	#           > A1.png
	#           > A2.png
	#           > A3.png
	#           > ...
	#       > UI/
	#           > killfeed.png
	#

	vidcap = loadVideo(input_filepath);
	totalFrames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
	FPS = int(vidcap.get(cv2.CAP_PROP_FPS))
	startFrame = dateToFrame(startTime, FPS)
	endFrame = dateToFrame(endTime, FPS)

	if endFrame < 0: # if the date was -mm:ss ,this implies a timestamp from the end.
		endFrame += totalFrames

	if startFrame >= totalFrames:
		raise ValueError("Starting time ("+startTime+") is greater than the video duration ("+frameToDate(totalFrames)+")")
		
	if startFrame < 0:
		raise ValueError("Starting time ("+startTime+") cannot be negative")
		
	if endFrame >= totalFrames:
		raise ValueError("Ending time ("+endTime+") is greater than the video duration ("+frameToDate(totalFrames)+")")
		
	if endFrame < 0:
		raise ValueError("Ending time ("+endTime+" -> "+frameToDate(endFrame)+") cannot be negative")

	def loadCroppedVideo(path, startFrame, endFrame):
		vidcap = loadVideo(path)
		vidcap.set(cv2.CAP_PROP_FRAME_COUNT, endFrame)
		vidcap.set(cv2.CAP_PROP_POS_FRAMES, startFrame)
		return vidcap

	maskCache = loadMaskCache(input_filepath)

	import sqlite3
	DB = sqlite3.connect("data/processed/MetaWatch_statistics.sqlite3")
	c = DB.cursor()
	c.execute("SELECT analysis_UUID FROM analysis")
	c.close()
	DB.close()
	# assuming 60 fps
	FPSFACTOR = FPS / 60
	HEROES_LIMITER:int = int(floor(300*FPSFACTOR)) # 5 sec
	LIMITER_TEAMNAME:int = int(floor(300*FPSFACTOR)) # 5 sec
	LIMITER_HEALTH:int = int(floor(10*FPSFACTOR)) # 1/6 s , one frame every 10 frame counts.
	LIMITER_ULTIMATE:int = int(floor(10*FPSFACTOR)) # 1/6 sec
	LIMITER_NICKNAME:int = int(floor(300*FPSFACTOR)) # 5 sec
	LIMITER_KILLFEED:int = int(floor(int(60*2)*FPSFACTOR))# a kill stays up during 7 secs, so we cut the video in 3.5 sec chunks in order to have 3 readings per kill. 

	files = {}
	manager = multiprocessing.Manager()
	lock = manager.Lock()
	TEAMS = ["A", "B"]
	HEROES = range(1,7,1)
	# TODO upgradeable by using subprocesses instead of threads. See multiprocess.
	# TODO RESOLVE THE CRASH WHEN STARTING THOSE PROCESSES.
	with concurrent.futures.ProcessPoolExecutor() as executor:
		extractors = []

		if health or exportHealth:
			# reset the video
			extractors.append(HealthExtractor(
							vid_UUID=vid_UUID,
							analysis_UUID=analysis_UUID,
							vidpath=input_filepath,
							startFrame=startFrame,
							endFrame=endFrame,
							frameLimiter=LIMITER_HEALTH,
							extract=health,
							export=exportHealth,
							input_filepath=input_filepath,
							output_filepath=output_filepath,
							profiling=profiling))
		if ultimate or exportUltimate:
			# reset the video
			extractors.append(UltimateExtractor(
							vid_UUID=vid_UUID,
							analysis_UUID=analysis_UUID,
							vidpath=input_filepath,
							startFrame=startFrame,
							endFrame=endFrame,
							frameLimiter=LIMITER_ULTIMATE,
							extract=ultimate,
							export=exportUltimate,
							input_filepath=input_filepath,
							output_filepath=output_filepath,
							profiling=profiling))
		if nickname or exportNickname:
			# reset the video					
			extractors.append(NicknameExtractor(
							vid_UUID=vid_UUID,
							analysis_UUID=analysis_UUID,
							vidpath=input_filepath,
							startFrame=startFrame,
							endFrame=endFrame,
							frameLimiter=LIMITER_NICKNAME,
							extract=nickname,
							export=exportNickname,
							input_filepath=input_filepath,
							output_filepath=output_filepath,
							profiling=profiling))
		if heroes:
			raise NotImplementedError();
			
		if teamname or exportTeam:
			extractors.append(TeamnameExtractor(
							vid_UUID=vid_UUID,
							analysis_UUID=analysis_UUID,
							vidpath=input_filepath,
							startFrame=startFrame,
							endFrame=endFrame,
							frameLimiter=LIMITER_TEAMNAME,
							extract=teamname,
							export=exportTeam,
							input_filepath=input_filepath,
							output_filepath=output_filepath,
							profiling=profiling))
		if killfeed or exportKillfeed:
			extractors.append(KillfeedExtractor(
							vid_UUID=vid_UUID,
							analysis_UUID=analysis_UUID,
							vidpath=input_filepath,
							startFrame=startFrame,
							endFrame=endFrame,
							frameLimiter=LIMITER_KILLFEED,
							extract=killfeed,
							export=exportKillfeed,
							input_filepath=input_filepath,
							output_filepath=output_filepath,
							profiling=profiling))
		futures = [executor.submit(runExtractor, extractor, lock) for extractor in extractors]
		for future in futures:
			future.result()
		


