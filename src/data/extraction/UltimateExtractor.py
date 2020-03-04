from . import *
import cv2
import logging
from .GLOBALS import STAT_DB_PATH

class UltimateExtractor(BaseExtractor): # extracts the data of a specific hero, specified by the team and hero arguments
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.name = __name__

	def onAnalyzeFrame(self, frame, image):
		for team in ["A", "B"]:
			for hero in range(1,7,1):
				maskId = (team, hero, "Ultimate")
				mask = self.maskCache[maskId]
				croppedImg = cropImage(cv2.bitwise_and(image, mask), maskId, mask)
				if self.export:
					cv2.imwrite(self.output_filepath+"_ultimate_f"+str(frame)+"_t"+team+"_h"+str(hero)+".png", croppedImg)
				if self.extract:
					#line:List[str] = [str(frame)]
					heroUltimateDetected:bool = convertUltimateImgToValue(croppedImg)
					
					#line.append("".join(["\"",team,"\""]))
					#line.append(str(hero))
					#line.append("".join(["T" if heroUltimateDetected else "F"]))
					#writeCSVLine(self.output_file, line)

					data = {
						"frame": frame, 
						"player_UUID": hero,
						"team_UUID" : 1 if team == "A" else 2,
						"ultimate_state" : 1 if heroUltimateDetected else 0,
					}
					self.writeToDB(data)
		

	def writeToDB(self, data=None, force=False):		
		if self.profiling:
			return
		if data is not None: 
			self.outputBuffer.append(data);
		if (len(self.outputBuffer) >= self.bufferLimit or force):
			with self.lock:
				DB = sqlite3.connect("data/processed/MetaWatch_statistics.sqlite3")
				try:
					
					c = DB.cursor()
					# vid_UUID INTEGER NOT NULL,
					# analysis_UUID INTEGER NOT NULL,
					# frame INTEGER NOT NULL,
					# team_UUID INTEGER NOT NULL,
					# player_UUID INTEGER NOT NULL,
					# ultimate_state INTEGER NOT NULL,
					# frame INTEGER NOT NULL,
					
					while len(self.outputBuffer) > 0:
						data = self.outputBuffer.pop(0)
						c.execute(f"""
						INSERT INTO RAW_ultimate(
							vid_UUID, 
							analysis_UUID, 
							frame, 
							player_UUID, 
							team_UUID, 
							ultimate_state) 
						VALUES (
							{self.vid_UUID}, 
							{self.analysis_UUID}, 
							{data["frame"]}, 
							{data["player_UUID"]}, 
							{data["team_UUID"]}, 
							{data["ultimate_state"]})
						""")
					
					c.close()
				
				except Exception as e:
					self.logger.error(e)
				finally:
					DB.commit()
					DB.close()