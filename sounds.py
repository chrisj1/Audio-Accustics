import pyaudio
import numpy as np


p = pyaudio.PyAudio()

volume = 0.5     # range [0.0, 1.0]
fs = 44100       # sampling rate, Hz, must be integer

# for paFloat32 sample values must be in range [-1.0, 1.0]
stream = p.open(format=pyaudio.paFloat32,
				channels=1,
				rate=fs,
			output=True)

def playSin(frequency):
	duration = 5000   # in seconds, may be float
	# generate samples, note conversion to float32 array
	samples = (np.sin(2*np.pi*np.arange(fs*duration)*frequency/fs)).astype(np.float32)
	# play. May repeat with different volume values (if done interactively) 
	stream.write(volume*samples)

playSin(60)

stream.stop_stream()
stream.close()

p.terminate()