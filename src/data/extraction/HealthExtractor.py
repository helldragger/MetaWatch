from . import *
import cv2
from .GLOBALS import STAT_DB_PATH
import logging
import threading

class HealthExtractor(BaseExtractor): # extracts the health data of a specific hero, specified by the team and hero arguments
	def __init__(self, *args, **kwargs):    
		super().__init__(*args, **kwargs)
		self.name = __name__

	def onAnalyzeFrame(self, frame, image):
		for team in ["A", "B"]:
			for hero in range(1,7,1):
				maskId = (team, hero, "Health")
				mask = self.maskCache[maskId]
				croppedImg = cropImage(cv2.bitwise_and(image, mask), maskId, mask)
				if self.export:
					cv2.imwrite(self.output_filepath+"_health_f"+str(frame)+"_t"+team+"_h"+str(hero)+".png", croppedImg)
				if self.extract:
					heroHealthNotches = convertHeroHealthImgToNotches(croppedImg)
					heroHealthColors:Dict[str, int] = convertNotchesToColors(heroHealthNotches, croppedImg)
					heroHealthData:Dict[str, int] = convertColorsToData(heroHealthColors)
					
					data = {
						"frame" : frame,
						"team_UUID" : 1 if team == "A" else 2,
						"player_UUID" : hero,
						"health":heroHealthData["health"],
						"armor":heroHealthData["armor"],
						"shield":heroHealthData["shield"],
						"damage":heroHealthData["damage"]
					}

					self.writeToDB(data)

	def writeToDB(self, data=None, force=False):
		if self.profiling:
			return
		if data is not None: 
			self.outputBuffer.append(data);
		if (len(self.outputBuffer) >= self.bufferLimit or force):
			with self.lock:
				DB = sqlite3.connect(STAT_DB_PATH)
				try:
					c = DB.cursor()
					
					# vid_UUID INTEGER NOT NULL,
					# analysis_UUID INTEGER NOT NULL,
					# frame INTEGER NOT NULL,
					# team_UUID INTEGER NOT NULL,
					# player_UUID INTEGER NOT NULL,
					# health INTEGER NOT NULL,
					# shield INTEGER NOT NULL,
					# armor INTEGER NOT NULL,
					# damage INTEGER NOT NULL,
					while len(self.outputBuffer) > 0:
						data = self.outputBuffer.pop(0)
						c.execute(f"""
						INSERT INTO RAW_healthbars(
							vid_UUID, 
							analysis_UUID, 
							frame, 
							team_UUID, 
							player_UUID, 
							health, 
							shield, 
							armor, 
							damage) 
						VALUES (
							{self.vid_UUID}, 
							{self.analysis_UUID}, 
							{data["frame"]}, 
							{data["team_UUID"]}, 
							{data["player_UUID"]}, 
							{data["health"]}, 
							{data["shield"]}, 
							{data["armor"]}, 
							{data["damage"]})
						""")
						
					c.close()
				except Exception as e:
					self.logger.error(e)
				finally:			
					DB.commit()
					DB.close()