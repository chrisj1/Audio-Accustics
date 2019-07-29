import numpy as np
from scipy.io import wavfile
import os
import xml.etree.ElementTree as ET
from sys import argv
from datetime import datetime,timedelta
from multiprocessing import Pool
from time import sleep
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.cluster import MiniBatchKMeans, KMeans


WORKER_NUM = 6
TERMS = 30
K_CLUSTERS = 5

def parseTime(time):
	return datetime.strptime(time, '%Y-%m-%dT%H:%M:%S')

def computeWindow(window):
	return cepstrumCoefficents(window.windowedFile)

class Recording(object):
	windows = None
	starttime = None
	endtime = None
	sessionname = None
	file = None
	windowlength = None
	samplerate = None
	gps = None
	tempature = None

	def __init__(self, starttime, endtime, sessionname, file=None, windowlength=5, samplerate=288000, gps=None, tempature=None):
		self.starttime = parseTime(starttime)
		self.endtime = parseTime(endtime)
		self.sessionname = sessionname
		self.file = file
		self.windowlength = windowlength
		self.gps = gps
		self.tempature = tempature
		self.windows = []

	def window(self, pool):
		destination = os.path.dirname(self.file)+"/" + str(self.starttime).replace(' ','_') + "_{}_second_windows".format(self.windowlength)
		if not os.path.exists(destination):
			os.mkdir(destination)
		command = "ffmpeg -loglevel error -i {} -f wav -f segment -segment_time {} -y {}/clip\%03d.wav".format(self.file,self.windowlength,destination)
		print("Windowing...")
		#os.system(command)

		for filename in sorted(os.listdir(destination)):	
			id = int(filename[4:-4])
			timeoffset = id * self.windowlength
			time = self.starttime + timedelta(seconds=timeoffset)
			# print(id, timeoffset, time)
			window = Window(time, timeoffset, destination + "/" + filename, self.windowlength, self)
			self.windows.append(window)

		print("Computing cepstrum coefients with {} terms and with {} processes".format(TERMS, WORKER_NUM))
		assert(len(self.windows) == 361)
		cepstrumCoefficents = pool.map(computeWindow, self.windows)
		assert(len(cepstrumCoefficents) == 361)
		for i in range(len(cepstrumCoefficents)):
			self.windows[i].cepstrum = cepstrumCoefficents[i]



class Window(object):
	cepstrum = None
	time = None
	orginalAudioFile = None
	windowedFile = None
	windowLength = None
	timeoffset = None
	recording = None

	def __init__(self, time, timeoffset, windowedFile, windowLength, recording):
		self.time = time
		self.timeoffset = timeoffset
		self.windowedFile = windowedFile
		self.windowLength = windowLength
		self.recording = recording
		
	def computeCepstrumCoefficents(self):
		self.cepstrum = cepstrumCoefficents(self.windowedFile)



# This code takes in windowed wav files and maps them to their cepstrum coeefients. Perhaps change this to compute the 
# Mel-frequency cepstral coeeficients. This should help with analyzing the spectral features, but may have issues when trying
# to match temporal features.
def cepstrumCoefficents(filename):
	# read in the audio file
	file = open(filename, 'rb')
	fs, signal = wavfile.read(file)

	warnings.filterwarnings('ignore')

	# compute the power spectrums on each window
	powerspectrum = np.abs(np.fft.fft(signal, n=TERMS))**2
	# compute the cepstrum. These should be real, but due to floating point issues we ignore the imaginary values.
	cepstrum = np.real(np.fft.ifft(np.nan_to_num(powerspectrum),n=TERMS))
	assert(len(cepstrum) == TERMS)
	file.close()

	warnings.filterwarnings('always')

	return cepstrum


def readInFromDirectory(directoryPath, sessionname):
	files = {}
	pool = Pool(WORKER_NUM)
	for filename in os.listdir(directoryPath):
		if filename.endswith("xml"):
			starttime = None
			stoptime = None
			tree = ET.parse(directoryPath + "/" + filename)
			root = tree.getroot()
			for event in root.iter('WavFileHandler'):
				if next(iter(event.attrib.keys())) == "SamplingStartTimeLocal":
					starttime = event.attrib['SamplingStartTimeLocal']
				elif next(iter(event.attrib.keys())) == "SamplingStopTimeLocal":
					stoptime = event.attrib['SamplingStopTimeLocal']
			recording = Recording(starttime, stoptime, sessionname, file=directoryPath + "/" + filename[:-8]+ ".wav")
			files[filename[:-8]] = recording
			recording.window(pool)
	pool.close()
	return list(files.values())

recordings = readInFromDirectory(argv[1], argv[2])

windows = []
for r in recordings:
	for w in r.windows:
		windows.append(w)

# lets try some k-means clustering
data = []
for window in windows:
	data.append(window.cepstrum)
data = scale(data)
print(data)

k_means = KMeans(init='k-means++', k=5, n_init=10)
t0 = time.time()
k_means.fit(X)
t_batch = time.time() - t0
k_means_labels = k_means.labels_
k_means_cluster_centers = k_means.cluster_centers_
k_means_labels_unique = np.unique(k_means_labels)

print(k_means)
print(k_means_labels_unique)