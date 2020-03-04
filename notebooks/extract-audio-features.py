#%%
from pyAudioAnalysis import audioAnalysis
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
from pyAudioAnalysis import MidTermFeatures
import matplotlib.pyplot as plt
import subprocess
import numpy as np

#extract some audio

VIDEOFILE = "../data/raw/8/replay.mp4"
AUDIOFILE = "./extracted.wav"
FEATUREFILE = "./extracted.ft"

command = f"ffmpeg -i {VIDEOFILE} -vn {AUDIOFILE} -y"

subprocess.call(command, shell=True)

[Fs, x] = audioBasicIO.read_audio_file(AUDIOFILE)
x = audioBasicIO.stereo_to_mono(x)

midF, shortF, midFNames = MidTermFeatures.mid_feature_extraction(x,Fs, 0.1*Fs,0.05*Fs,0.05*Fs,0.025*Fs)

np.save(FEATUREFILE, midF)
np.savetxt(FEATUREFILE + ".csv", midF.T, delimiter=",", header=",".join(midFNames))
#%%
audioAnalysis.thumbnailWrapper(AUDIOFILE,50)
#explore the audio

audioAnalysis.fileSpectrogramWrapper(AUDIOFILE)

audioAnalysis.fileChromagramWrapper(AUDIOFILE)

audioAnalysis.beatExtractionWrapper(AUDIOFILE, True)
#%%
var = 48
print(f"{var} : {3/199:.5f}")
#%%