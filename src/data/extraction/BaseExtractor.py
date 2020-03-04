import cv2
import logging
import multiprocessing
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput
import sqlite3
from . import loadMaskCache, frameToDate
import time

def loadVideo(input_filepath:str):
	return cv2.VideoCapture(input_filepath+"/replay.mp4");

def loadCroppedVideo(path, startFrame, endFrame):
	vidcap = loadVideo(path)
	#vidcap.set(cv2.CAP_PROP_FRAME_COUNT, endFrame)
	vidcap.set(cv2.CAP_PROP_POS_FRAMES, startFrame)
	return vidcap

class BaseExtractor:
	def __init__(self, *args, **kwargs):

		self.vid_UUID = kwargs.pop('vid_UUID')
		self.analysis_UUID = kwargs.pop('analysis_UUID')
		# video related
		self.vidpath = kwargs.pop('vidpath') 
		self.startFrame = kwargs.pop('startFrame')
		self.endFrame = kwargs.pop('endFrame')

		self.frameLimiter = kwargs.pop('frameLimiter')
		if self.frameLimiter < 1:
			raise ValueError("frame limiter must be equal or greater than 1 (1 = every frame)")
		
		self.completedIterations = 0

		
		# Data extracting related
		self.extract = kwargs.pop('extract', False)
		self.output_file = kwargs.pop('output_file', None) # files["killfeed"]
		
		# Frame exporting related
		self.export = kwargs.pop('export', False)
		self.output_filepath = kwargs.pop('output_filepath') # for img exports

		# profiling related
		self.profiling = kwargs.pop("profiling", False)
		self.name = __name__

		self.input_filepath = kwargs.pop("input_filepath", False)

		# DB writing buffer related
		self.outputBuffer:[] = [];
		self.bufferLimit = 1000;

		# Logging related
		self.lastLoggingTime = time.monotonic();
		self.loggingInterval = 60 # in seconds


	def loadFrame(self):
		return self.vidcap.read()      

	def skipFrames(self, newFrame):
		if self.frameLimiter>1:
			self.vidcap.set(cv2.CAP_PROP_POS_FRAMES, newFrame)

	def onAnalyzeVid(self):
		self.vidcap = loadCroppedVideo(self.vidpath , self.startFrame,self.endFrame)
		self.FPS = int(self.vidcap.get(cv2.CAP_PROP_FPS))
		self.startDate = frameToDate(self.startFrame, self.FPS)
		self.endDate = frameToDate(self.endFrame, self.FPS)

		self.totalDate = frameToDate(self.endFrame - self.startFrame, self.FPS)
		self.totalIterations = (self.endFrame - self.startFrame) // self.frameLimiter
		self.logger.info("VIDEO LOADED")
		frame = int(self.vidcap.get(cv2.CAP_PROP_POS_FRAMES))
		self.lastCompletedIterations = 0
		self.lastLoggedFrame = frame
		self.logger.info(f"STARTING ANALYSIS | vid {self.vid_UUID} | analysis {self.analysis_UUID} | {self.FPS} FPS | start {self.startDate} --> {self.endDate}")

		success , image = self.loadFrame()

		if not success:
			raise Exception("couldn't read video frame "+str(frame))
		# TODO upgrade the extraction using multiple threads.
		
		while success and frame <= self.endFrame:
			self.logProgress(frame)

			self.onAnalyzeFrame(frame, image)
			self.completedIterations += 1
			frame += self.frameLimiter
			self.skipFrames(frame)
			success , image = self.loadFrame()
		self.logComplete()

	def logProgress(self, frame):
		currTime = time.monotonic()
		if (currTime - self.lastLoggingTime) >= self.loggingInterval:
			# iterations
			iterationRemaining = self.totalIterations - self.completedIterations
			# iteration.s-1
			iterationSpeed = (self.completedIterations - self.lastCompletedIterations) / (currTime - self.lastLoggingTime)
			# iteration.(iteration.s-1)-1 = s
			ETA = (iterationRemaining / iterationSpeed )
			# we consider a s as a frame to simplify the view:
			self.logger.info(f" {frameToDate(frame - self.startFrame, self.FPS)} / {self.totalDate} | {round(self.completedIterations/self.totalIterations *100, 1)} % | {frameToDate(ETA, FPS=1)} remaining | {self.startDate} -- {frameToDate(frame, self.FPS)} -> {self.endDate} ")
			self.lastLoggingTime = currTime
			self.lastCompletedIterations = self.completedIterations

	def logComplete(self):
		self.logger.info("Extraction complete")

	def setup(self, lock):
		log_fmt = '[%(asctime)s] [%(name)s] [%(levelname)s] : %(message)s'
		logging.basicConfig(level=logging.INFO, format=log_fmt)

		# Logging related
		self.logger = logging.getLogger(self.name)
		from logging import FileHandler
		self.logger.addHandler(FileHandler(f'logs/{self.name}.log', mode="a"))
		self.logger.setLevel(logging.INFO)

		self.lock = lock
		
		self.maskCache = loadMaskCache(self.input_filepath)
		
	def run(self):
		try:
			if self.profiling:
				self.logger.warn("PROFILING MODE, EXTRACTED DATA WILL NOT BE SAVED.")
				graphviz = GraphvizOutput()
				graphviz.output_file = "logs/"+self.name+'_profile.png'
				with PyCallGraph(output=graphviz):
					self.onAnalyzeVid()
			else:
				self.onAnalyzeVid()
				self.writeToDB(force=True)
		except Exception as e:
			self.logger.error(e)
		
	def writeToDB(self, data=None, force=False):
		raise NotImplementedError
