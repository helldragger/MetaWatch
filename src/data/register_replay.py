import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import sqlite3
from GLOBALS import FILE_DB_PATH, STAT_DB_PATH
import os
from shutil import copyfile
import json
from datetime import date
def generateDB():
	logger = logging.getLogger(__name__)
	logger.info(f'Generating DB @{STAT_DB_PATH}')
	qry = open('src/data/create_MW_stat_db.sql', 'r').read()
	DB = sqlite3.connect(STAT_DB_PATH)
	c = DB.cursor()
	c.executescript(qry)
	DB.commit()
	DB.close()

def registerVid(video_filepath):
	logger = logging.getLogger(__name__)
	logger.info(f'Registering video @{video_filepath}')
	try:
		
		DB = sqlite3.connect(STAT_DB_PATH)
		c = DB.cursor()

		if not os.path.isfile(video_filepath):
			raise FileNotFoundError(f"[Errno 2] No such file: '{video_filepath}'")

		if not video_filepath.endswith(".mp4"):
			raise ValueError("The video must be an .mp4 file")

		last_vid_UUID = c.execute("SELECT vid_UUID FROM videos ORDER BY vid_UUID desc LIMIT 1").fetchone()
		if last_vid_UUID is None:
			# if no videos are currently registered, then we put the first UUID at 1
			last_vid_UUID = 1
		else:
			# else we just increment 
			last_vid_UUID = last_vid_UUID[0] + 1
		
		# first we try to discover if there isn't already another video at this UUID place 
		while os.path.isdir(FILE_DB_PATH + str(last_vid_UUID)):
			# then there already is a video we haven't registered yet.
			# so we register it.
			c.execute(f"INSERT INTO videos(vid_UUID) VALUES ({last_vid_UUID})")
			logger.info(f"Already present but unknown video succesfully registered as {last_vid_UUID}")
			last_vid_UUID += 1
		
		#now there is actually no existing folder of this name
		# we create it and copy the video in it.		

		os.mkdir(FILE_DB_PATH + str(last_vid_UUID))
		logger.info(f"Video folder succesfully created")
		copyfile(video_filepath, FILE_DB_PATH + str(last_vid_UUID)+"/replay.mp4")

		logger.info(f"Video succesfully copied into the database")
		# we write down some metadata from what the file was called before being renamed to replay.mp4
		with open(FILE_DB_PATH + str(last_vid_UUID)+"/about.txt", "w") as metadata:
			json.dump({"ID":last_vid_UUID,"name":video_filepath,"added":date.today().strftime("%d/%m/%Y")}, metadata)

		logger.info(f"Video metadata succesfully added to the database")
		# TODO generate preview
		# we then register the video as "in the database"
		c.execute(f"INSERT INTO videos(vid_UUID) VALUES ({last_vid_UUID})")
		logger.info(f"Video succesfully registered as {last_vid_UUID}")
	except Exception as err:
		logger.exception("The program encountered a critical error and will now quit.")
	finally:
		DB.commit()
		DB.close()

@click.command()
@click.argument('video_filepath', type=click.Path(exists=False))
def main(video_filepath:str):
	""" 
	Registers and moves a video into the video database
	"""
	logger = logging.getLogger(__name__)
	logger.info(f'Registering video @{video_filepath}')

	if not os.path.isfile(STAT_DB_PATH):
		generateDB()

	if os.path.isdir(video_filepath):
		onlyVids = [os.path.join(video_filepath, f) for f in os.listdir(video_filepath) if os.path.isfile(os.path.join(video_filepath, f)) and os.path.join(video_filepath, f).endswith(".mp4")]
		for vid in onlyVids:
			registerVid(vid)
	else:
		registerVid(video_filepath)
	


if __name__ == '__main__':
	log_fmt = '[%(asctime)s] [%(name)s] [%(levelname)s] : %(message)s'
	logging.basicConfig(level=logging.INFO, format=log_fmt)

	# not used in this stub but often useful for finding various files
	project_dir = Path(__file__).resolve().parents[2]

	# find .env automagically by walking up directories until it's found, then
	# load up the .env entries as environment variables
	load_dotenv(find_dotenv())

	main()

