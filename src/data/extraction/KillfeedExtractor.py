from . import *
import cv2
import logging
from .GLOBALS import STAT_DB_PATH

from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color

class KillfeedExtractor(BaseExtractor):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.name = __name__
	
	def initializeTeamColors(self):
		self.maskId = ("UI", "killfeed")
		self.mask = self.maskCache[self.maskId]

		frame_temp = self.vidcap.get(cv2.CAP_PROP_POS_FRAMES)
		success, image = self.vidcap.read()

		teamAMaskId = ("A", 1, "Ultimate")
		teamAMask = self.maskCache[teamAMaskId]
		teamAImg = cropImage(cv2.bitwise_and(image, teamAMask), teamAMaskId, teamAMask) 
		# this should give the ultimate number crop, which has a little border of the team color as padding
		# to avoid the image X shearing due to the team, we center our pixel on the X axis
		# we then use the first pixel of the vertical axis to get the padding color.
		
		self.team_A_color = sampleColor(teamAImg, (0, teamAImg.shape[1]//2), "Team A color sampling")
		self.logger.info("Detected team A primary color:"+str(self.team_A_color))
		# assuming a 0.
		team_A_secondary_color = sampleColor(teamAImg, (3, int(teamAImg.shape[1]*0.75)), "Team A secondary color sampling")
		self.logger.info("Detected team A secondary color:"+str(team_A_secondary_color))
			

		#same here
		teamBMaskId = ("B", 1, "Ultimate")
		teamBMask = self.maskCache[teamBMaskId]
		teamBImg = cropImage(cv2.bitwise_and(image, teamBMask), teamBMaskId, teamBMask)
		
		self.team_B_color = sampleColor(teamBImg, (0, teamBImg.shape[1]//2), "Team B color sampling")
		self.logger.info("Detected team B color:"+str(self.team_B_color))
		team_B_secondary_color = sampleColor(teamBImg, (3, int(teamBImg.shape[1]*0.75)), "Team B secondary color sampling")
		self.logger.info("Detected team B secondary color:"+str(team_B_secondary_color))
		
		# we reset the video at the first expected frame.
		self.vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_temp)

		# we also save the team colors into a specific file for later theming.
		with open(self.output_filepath+"_teamcolors.csv", "w") as color_output_file:
			color_output_file.write(",".join(["team","type","RED", "GREEN", "BLUE"]))
			writeCSVLine(color_output_file, ["A","PRIMARY",str(self.team_A_color[2]), str(self.team_A_color[1]), str(self.team_A_color[0])])
			writeCSVLine(color_output_file, ["B","PRIMARY",str(self.team_B_color[2]), str(self.team_B_color[1]), str(self.team_B_color[0])])
			# TODO make the detection more robust by clustering the colors into two categories depending on cluster size.
			# assuming a 0.
			writeCSVLine(color_output_file, ["A","SECONDARY",str(team_A_secondary_color[2]), str(team_A_secondary_color[1]), str(team_A_secondary_color[0])])
			writeCSVLine(color_output_file, ["B","SECONDARY",str(team_B_secondary_color[2]), str(team_B_secondary_color[1]), str(team_B_secondary_color[0])])
		
		# here we prepare the team colors for later comparisons in the cieLAB colorspace.
		self.team_A_color = sRGBColor(self.team_A_color[2]/255, self.team_A_color[1]/255, self.team_A_color[0]/255)
		self.team_A_color = convert_color(self.team_A_color, LabColor);
		
		self.team_B_color = sRGBColor(self.team_B_color[2]/255, self.team_B_color[1]/255, self.team_B_color[0]/255)
		self.team_B_color = convert_color(self.team_B_color, LabColor);
	
	def onAnalyzeFrame(self, frame, image):
		if "team_A_color" not in self.__dict__: 
			self.initializeTeamColors()

		killfeedImg = cropImage(cv2.bitwise_and(image, self.mask), self.maskId, self.mask)   

		if self.export:          
			cv2.imwrite(self.output_filepath+"_killfeed_f"+str(frame)+".png", killfeedImg)    
		if self.extract:
			feed = readKillFeed(killfeedImg, kill_arrow_img, self.team_A_color, self.team_B_color)
			for killData in feed:
				#line:List[str] = [str(frame)]
				#line.append("\""+killData["killed"]+"\"")
				#line.append("\""+killData["killed.team"]+"\"")
				#line.append("\""+killData["killer"]+"\"")
				#line.append("\""+killData["killer.team"]+"\"")
				#line.append("\""+killData["assists"]+"\"")
				#line.append("\""+killData["ability"]+"\"")
				#line.append("T" if killData["is.crit"] == "True" else "F")
				#writeCSVLine(self.output_file, line)

				data = {
					"frame": frame, 
					"is_crit": 1 if killData["is.crit"] == "True" else 0,
					"killed_hero_name" : killData["killed"],
					"killed_hero_team_UUID" : 1 if killData["killed.team"] == "A" else  (2 if killData["killed.team"] == "B" else -1),
					"killer_hero_name" : killData["killer"],
					"killer_hero_team_UUID" : 1 if killData["killer.team"] == "A" else  (2 if killData["killer.team"] == "B" else -1),
					"assists_names" : killData["assists"],
					"ability_key_UUID": "NULL" if killData["ability"] == "none" else  killData["ability"],
					"match_coordinates": killData["match_coordinates"]
				}
				self.writeToDB(data)
	
	
	def writeToDB(self, data=None,force=False):
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
					# is_crit INTEGER NOT NULL,
					# killed_hero_name TEXT NOT NULL,
					# killed_hero_team_UUID INTEGER NOT NULL,
					# killer_hero_name TEXT NOT NULL,
					# killer_hero_team_UUID INTEGER NOT NULL,
					# assists_names TEXT NOT NULL,
					# ability_key_UUID INTEGER NOT NULL,
					#   match_coordinates TEXT NOT NULL
					while len(self.outputBuffer) > 0:
						data = self.outputBuffer.pop(0)
						self.logger.debug(data)
						c.execute(f"""
						INSERT INTO RAW_killfeed(
							vid_UUID, 
							analysis_UUID, 
							frame, 
							is_crit, 
							killed_hero_name, 
							killed_hero_team_UUID, 
							killer_hero_name, 
							killer_hero_team_UUID, 
							assists_names,
							ability_key_UUID,
							match_coordinates) 
						VALUES (
							{self.vid_UUID}, 
							{self.analysis_UUID}, 
							{data["frame"]}, 
							{data["is_crit"] }, 
							'{data["killed_hero_name"]}', 
							{data["killed_hero_team_UUID"]}, 
							'{data["killer_hero_name"]}', 
							{data["killer_hero_team_UUID"]}, 
							'{data["assists_names"]}', 
							{data["ability_key_UUID"]}, 
							'{data["match_coordinates"]}')
						""")
					c.close()
				except Exception as e:
					self.logger.error(e)
				finally:			
					DB.commit()
					DB.close()