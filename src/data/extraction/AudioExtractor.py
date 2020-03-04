import moviepy.editor as mp
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioAnalysis
from pyAudioAnalysis import ShortTermFeatures
from pyAudioAnalysis import MidTermFeatures
import numpy as np
import matplotlib.pyplot as plt

def extractAudio(vid_uuid, startTime, endTime):
    clip = mp.VideoFileClip("../../../data/raw/"+ str(vid_uuid) + "/replay.mp4").subclip(startTime, endTime)
    clip.audio.write_audiofile("../../../data/raw/"+ str(vid_uuid) + "/audio.wav")

def analysisAudio(vid_uuid):
    VIDEOFILE = "../../../data/raw/"+ str(vid_uuid) + "/replay.mp4"
    AUDIOFILE = "../../../data/raw/"+ str(vid_uuid) + "/audio.wav"
    FEATUREFILE = "../../../data/processed/"+ str(vid_uuid) + "_extracted.ft"
    [Fs, x] = audioBasicIO.read_audio_file(AUDIOFILE)
    x = audioBasicIO.stereo_to_mono(x)

    midF, shortF, midFNames = MidTermFeatures.mid_feature_extraction(x,Fs, (1/30)*Fs,(1/60)*Fs,(1/60)*Fs,(1/120)*Fs)

    np.save(FEATUREFILE, midF)
    np.savetxt(FEATUREFILE + ".csv", midF.T, delimiter=",", header=",".join(midFNames))
    #%%
    audioAnalysis.thumbnailWrapper(AUDIOFILE,50)
    #explore the audio
    #audioAnalysis.fileSpectrogramWrapper(AUDIOFILE)
    #audioAnalysis.fileChromagramWrapper(AUDIOFILE)


extractAudio(31,4024,4350)
analysisAudio(31)
