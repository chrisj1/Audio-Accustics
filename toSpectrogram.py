import os
import wave

import pylab
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plot


def graph_spectrogram(directory,filename):
	samplingFrequency, signalData = wavfile.read(directory + filename +'.wav')

	# Plot the signal read from wav file

	plot.subplot(211)

	plot.title('Spectrogram')

	 

	plot.plot(signalData)

	plot.xlabel('Sample')

	plot.ylabel('Amplitude')


	plot.subplot(212)

	plot.specgram(signalData,Fs=samplingFrequency)

	plot.xlabel('Time')

	plot.ylabel('Frequency')

	plot.savefig(filename+'.png', bbox_inches='tight', dpi='figure', quality=100)



directory = "/Users/chrisjerrett/Desktop/research/1minuteclips/"

for filename in os.listdir(directory):
	if filename.endswith(".wav"): 
		print(filename)
		graph_spectrogram(directory,filename[:-4])