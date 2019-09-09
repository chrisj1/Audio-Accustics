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
import SimpSOM as sps
from tf_som import SelfOrganizingMap
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import logging
from mpl_toolkits.mplot3d import Axes3D
import somoclu
import h5py

WORKER_NUM = 12
TERMS = 200
EPOCHS = 2
IMAGE_SIZE = 2**16-1
def parseTime(time):
	return datetime.strptime(time, '%Y-%m-%dT%H:%M:%S')

def computeWindow(window):
	return  cepstrumCoefficents(window.windowedFile)

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

	def window(self, pool, number, total):
		destination = os.path.dirname(self.file)+"/" + str(self.starttime).replace(' ','_') + "_{}_second_windows".format(self.windowlength)
		if not os.path.exists(destination):
			os.mkdir(destination)
		command = "ffmpeg -loglevel error -i {} -f wav -f segment -segment_time {} -y {}/clip\%03d.wav -threads 10".format(self.file,self.windowlength,destination)
		print("Windowing with start 0...")
		if(not len(argv) > 3 or argv[3] != 'false'):
			os.system(command)

		for filename in sorted(os.listdir(destination)):
			id = int(filename[4:-4])
			timeoffset = id * self.windowlength
			time = self.starttime + timedelta(seconds=timeoffset)
			# print(id, timeoffset, time)
			window = Window(time, timeoffset, destination + "/" + filename, self.windowlength, self)
			self.windows.append(window)

		destination = os.path.dirname(self.file)+"/" + str(self.starttime).replace(' ','_') + "_{}_second_offset_windows".format(self.windowlength)
		if not os.path.exists(destination):
			os.mkdir(destination)
		command = "ffmpeg -ss {} -loglevel error -i {} -f wav -f segment -segment_time {} -y {}/clip\%03d.wav -threads 10".format(self.windowlength/2, self.file,self.windowlength,destination)
		print("Windowing with start {}...".format(self.windowlength/2))
		if(not len(argv) > 3 or argv[3] != 'false'):
			os.system(command)

		for filename in sorted(os.listdir(destination)):
			id = int(filename[4:-4])
			timeoffset = id * self.windowlength
			time = self.starttime + timedelta(seconds=timeoffset)
			# print(id, timeoffset, time)
			window = Window(time, timeoffset, destination + "/" + filename, self.windowlength, self)
			self.windows.append(window)

		print("{}/{} Computing cepstrum coefients with {} terms and with {} processes".format(number, total, TERMS, WORKER_NUM))

		cepstrumCoefficents = pool.map(computeWindow, self.windows)
	
		for i in range(len(cepstrumCoefficents)):
			self.windows[i].cepstrum = cepstrumCoefficents[i][0]

class Window(object):
	cepstrum = None
	time = None
	orginalAudioFile = None
	windowedFile = None
	windowLength = None
	timeoffset = None
	recording = None
	max_volume = None
	min_volume = None
	mean_volume = None
	sample_rate = None

	def __init__(self, time, timeoffset, windowedFile, windowLength, recording):
		self.time = time
		self.timeoffset = timeoffset
		self.windowedFile = windowedFile
		self.windowLength = windowLength
		self.recording = recording

	def computeCepstrumCoefficents(self):
		c, f = cepstrumCoefficents(self.windowedFile)
		self.cepstrum = c
		self.sample_rate = f


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

	return cepstrum, fs


def readInFromDirectory(directoryPath, sessionname):
	files = {}
	pool = Pool(WORKER_NUM)
	total = len(os.listdir(directoryPath))
	number = 0
	for filename in os.listdir(directoryPath):
		number+=1
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
			recording.window(pool, number, total)
	pool.close()
	return list(files.values())

recordings = readInFromDirectory(argv[1], argv[2])

windows = []
labels = []
for r in recordings:
	for w in r.windows:
		windows.append(w)
		labels.append(w.windowedFile)

print("Building dataset")
data = np.empty([len(windows),TERMS])

for i in range(len(windows)):
	for j in range(len(windows[i].cepstrum)):
		data[i][j] = windows[i].cepstrum[j]

print("Reshaping Data")
data = data.reshape((len(windows), TERMS))
print("Reshaped data")
print(data.shape)

print("Start training")
n_rows, n_columns = 100, 160
som = somoclu.Somoclu(n_columns, n_rows,
	compactsupport=True, initialization="pca", gridtype='hexagonal', kerneltype=0, verbose=2)
print("Running {} epochs".format(EPOCHS))
som.train(data, epochs=EPOCHS)
print("done traiing")
som.cluster()
print("done clustering")
print("Saving data")

bmus = h5py.File('bmus.h5', 'w')
bmus.create_dataset('bmus', data=som.bmus)

umatrix = h5py.File('umatrix.h5', 'w')
umatrix.create_dataset('umatrix', data=som.umatrix)

codebook = h5py.File('codebook.h5', 'w')
codebook.create_dataset('codebook', data=som.codebook)

som.view_umatrix(bestmatches=True, labels=labels, filename="umatrix_best.png", figsize=(IMAGE_SIZE, IMAGE_SIZE))
som.view_similarity_matrix(bestmatches=True, labels=labels, filename="similarity_best.png", figsize=(IMAGE_SIZE, IMAGE_SIZE))
som.view_component_planes(bestmatches=True, labels=labels, filename="component_best.png", figsize=(IMAGE_SIZE, IMAGE_SIZE))
som.view_activation_map(bestmatches=True, labels=labels, filename="activation_best.png", figsize=(IMAGE_SIZE, IMAGE_SIZE))

som.view_umatrix(bestmatches=False, labels=labels, filename="umatrix.png", figsize=(IMAGE_SIZE, IMAGE_SIZE))
som.view_similarity_matrix(bestmatches=False, labels=labels, filename="similarity.png", figsize=(IMAGE_SIZE, IMAGE_SIZE))
som.view_component_planes(bestmatches=False, labels=labels, filename="component.png", figsize=(IMAGE_SIZE, IMAGE_SIZE))
som.view_activation_map(bestmatches=False, labels=labels, filename="activation.png", figsize=(IMAGE_SIZE, IMAGE_SIZE))

som.view_umatrix(bestmatches=True, filename="umatrix_best_no.png", figsize=(IMAGE_SIZE, IMAGE_SIZE))
som.view_similarity_matrix(bestmatches=True, filename="similarity_best_no.png", figsize=(IMAGE_SIZE, IMAGE_SIZE))
som.view_component_planes(bestmatches=True, filename="component_best_no.png", figsize=(IMAGE_SIZE, IMAGE_SIZE))
som.view_activation_map(bestmatches=True, filename="activation_best_no.png", figsize=(IMAGE_SIZE, IMAGE_SIZE))

som.view_umatrix(bestmatches=False, filename="umatrix_no.png", figsize=(IMAGE_SIZE, IMAGE_SIZE))
som.view_similarity_matrix(bestmatches=False, filename="similarity_no.png", figsize=(IMAGE_SIZE, IMAGE_SIZE))
som.view_component_planes(bestmatches=False, filename="component_no.png", figsize=(IMAGE_SIZE, IMAGE_SIZE))
som.view_activation_map(bestmatches=False, filename="activation_no.png", figsize=(IMAGE_SIZE, IMAGE_SIZE))