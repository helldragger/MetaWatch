# -*- coding: utf-8 -*-
#import cython
#import pyximport; pyximport.install(pyimport=True, load_py_module_on_import_failure=True);

import replayAnalyzer as replayAnalyzer
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import sqlite3
from GLOBALS import FILE_DB_PATH, STAT_DB_PATH

@click.command()
@click.argument('vid_uuid', type=click.INT)
#@click.argument('output_filepath', type=click.Path())
@click.option('--profiling', default=False, is_flag=True)
@click.option('--health', default=False, is_flag=True)
@click.option('--ultimate', default=False, is_flag=True)
@click.option('--nickname', default=False, is_flag=True)
@click.option('--heroes', default=False, is_flag=True)
@click.option('--teamname', default=False, is_flag=True)
@click.option('--killfeed', default=False, is_flag=True)
@click.option('--all', default=False, is_flag=True)
@click.option('--export-health-img', default=False, is_flag=True)
@click.option('--export-ultimate-img', default=False, is_flag=True)
@click.option('--export-nickname-img', default=False, is_flag=True)
@click.option('--export-team-img', default=False, is_flag=True)
@click.option('--export-heroes-img', default=False, is_flag=True)
@click.option('--export-killfeed-img', default=False, is_flag=True)
@click.option('--start-timestamp', default="00:00:00", is_flag=False)
@click.option('--end-timestamp', default="-00:00:00", is_flag=False)
def main(vid_uuid, profiling, health, ultimate, nickname, heroes, teamname, killfeed, all, export_health_img, export_ultimate_img, export_nickname_img, export_team_img, export_heroes_img, export_killfeed_img, start_timestamp, end_timestamp):
	""" Runs data processing scripts to turn raw data from (../raw) into
		cleaned data ready to be analyzed (saved in ../processed).
	"""
	health = health or all
	ultimate = ultimate or all
	nickname = nickname or all
	teamname = teamname or all
	killfeed = killfeed or all
	logger = logging.getLogger(__name__)
	try:		
		DB = sqlite3.connect(STAT_DB_PATH)
		c = DB.cursor()

		# check if the video is registered
		vid_UUID_registered = c.execute(f"SELECT vid_UUID FROM videos WHERE vid_UUID={vid_uuid} LIMIT 1").fetchone() is not None
		if not vid_UUID_registered:
			raise FileNotFoundError(f"No such replay UUID registered in the database: '{vid_uuid}'")

		last_analysis_UUID = c.execute("SELECT analysis_UUID FROM analysis ORDER BY analysis_UUID desc LIMIT 1").fetchone()
		if last_analysis_UUID is None:
			# if no videos are currently registered, then we put the first UUID at 1
			analysis_UUID = 1
		else:
			# else we just increment 
			analysis_UUID = last_analysis_UUID[0] + 1
		# then we register this analysis on this video.
		c.execute(f"INSERT INTO analysis(analysis_UUID, vid_UUID) VALUES ({analysis_UUID},{vid_uuid})")
		DB.commit()

		
		# detect if the database is up and running, and most importantly available
		input_filepath = FILE_DB_PATH + str(vid_uuid) + "/";
		output_filepath = "data/processed/"+str(vid_uuid)
		logger.info('Starting replay analysis')
		replayAnalyzer.convertingVideoToImages2(input_filepath, output_filepath, profiling, health, ultimate, nickname,  heroes, teamname, killfeed, export_health_img, export_ultimate_img, export_nickname_img, export_team_img, export_heroes_img, export_killfeed_img, start_timestamp, end_timestamp, vid_uuid, analysis_UUID);
		logger.info("Analysis finished")
	except Exception as err:
		logger.exception("The program encountered a critical error and will now quit.")
		quit(-1)


if __name__ == '__main__':
	log_fmt = '[%(asctime)s] [%(name)s] [%(levelname)s] : %(message)s'
	logging.basicConfig(level=logging.INFO, format=log_fmt)

	# not used in this stub but often useful for finding various files
	project_dir = Path(__file__).resolve().parents[2]

	# find .env automagically by walking up directories until it's found, then
	# load up the .env entries as environment variables
	load_dotenv(find_dotenv())

	main()
