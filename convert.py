# ffmpeg -i  name_offiele -f segment -segment_time 3 -c copy out%03d.mp3

import os
directory = "/Users/chrisjerrett/Desktop/sounds/0611-061219SawmillBay5MOvernight"

def split(file):
	print("ffmpeg -i  {}/{}.wav -f wav -f segment -segment_time 60 -y 1minuteclips/{}_number\%03d.wav".format(directory,file,file))
	os.system("ffmpeg -i  {}/{}.wav -f wav -f segment -segment_time 60 -y 1minuteclips/{}_number\%03d.wav".format(directory, file,file))

	print("ffmpeg -i  {}/{}.wav -f mp3 -f segment -segment_time 60 -y 1minuteclips/{}_number\%03d.wav".format(directory,file,file))
	os.system("ffmpeg -i  {}/{}.wav -f wav -f segment -segment_time 5 -y 5secondclips/{}_number\%03d.wav".format(directory, file,file))


for filename in os.listdir(directory):
	if filename.endswith(".wav"): 
		print(filename[:-4])
		split(filename[:-4])