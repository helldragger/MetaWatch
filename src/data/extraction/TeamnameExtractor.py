from . import *
import cv2
import logging
from .GLOBALS import STAT_DB_PATH

class TeamnameExtractor(BaseExtractor): # extracts the data of a specific hero, specified by the team and hero arguments
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.name = __name__

	def onAnalyzeFrame(self, frame, image):
		for team in ["A", "B"]:
			
			maskId = (team, "Team")                
			mask = self.maskCache[maskId]

			croppedImg = cropImage(cv2.bitwise_and(image, mask), maskId, mask)
			if self.export:
				cv2.imwrite(self.output_filepath+"_team_f"+str(frame)+"_t"+team+".png", croppedImg)
			if self.extract:
				#line:List[str] = [str(frame)]
				teamText = convertImgToTeamname(croppedImg)
				#line.append("".join(["\"",team,"\""]))
				#line.append("\""+teamText+"\"")
				#writeCSVLine(self.output_file, line)

				data = {
					"frame": frame, 
					"team_UUID" : 1 if team == "A" else  2,
					"team_name" : teamText
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
					# player_UUID 
					# team_UUID 
					# player_name 
					while len(self.outputBuffer) > 0:
						data = self.outputBuffer.pop(0)
						c.execute(f"""
						INSERT INTO RAW_team_names(
							vid_UUID, 
							analysis_UUID, 
							frame,  
							team_UUID, 
							team_name) 
						VALUES (
							{self.vid_UUID}, 
							{self.analysis_UUID}, 
							{data["frame"]}, 
							{data["team_UUID"]}, 
							'{data["team_name"]}')
						""")
					
					c.close()
				except Exception as e:
					self.logger.error(e)
				finally:			
					DB.commit()
					DB.close()