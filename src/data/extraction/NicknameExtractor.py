from . import *
import cv2
import logging
from .GLOBALS import STAT_DB_PATH

class NicknameExtractor(BaseExtractor): # extracts the data of a specific hero, specified by the team and hero arguments
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.name = __name__

	def onAnalyzeFrame(self, frame, image):
		for team in ["A", "B"]:
			for hero in range(1,7,1):
				maskId = (team, hero, "Nickname")
				mask = self.maskCache[maskId]
				croppedImg = cropImage(cv2.bitwise_and(image, mask), maskId, mask)
				if self.export:
					cv2.imwrite(self.output_filepath+"_nickname_f"+str(frame)+"_t"+team+"_h"+str(hero)+".png", croppedImg)
				
				if self.extract:
					#line:List[str] = [str(frame)]

					heroNicknameText:str = ""
					heroNicknameText = convertImgToText(croppedImg)
					
					#line.append("".join(["\"",team,"\""]))
					#line.append(str(hero))
					#line.append("".join(["\"",heroNicknameText,"\""]))
					#writeCSVLine(self.output_file, line)

					data = {
						"frame": frame, 
						"player_UUID": hero,
						"team_UUID" : 1 if team == "A" else  2,
						"player_name" : heroNicknameText
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
						INSERT INTO RAW_players_nicknames(
							vid_UUID, 
							analysis_UUID, 
							frame, 
							player_UUID, 
							team_UUID, 
							player_name) 
						VALUES (
							{self.vid_UUID}, 
							{self.analysis_UUID}, 
							{data["frame"]}, 
							{data["player_UUID"]}, 
							{data["team_UUID"]}, 
							'{data["player_name"]}')
						""")
					
					c.close()
				except Exception as e:
					self.logger.error(e)
				finally:			
					DB.commit()
					DB.close()