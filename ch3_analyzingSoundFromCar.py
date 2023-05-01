from scipy.io import wavfile

import numpy as np

import matplotlib.pyplot as plt

AUDIO_FILE = "data\sound_sample_car_engine.wav"

sampling_freq, sound_data = wavfile.read(AUDIO_FILE)

print("Sampling Frequency = ", sampling_freq, "\nShape of data array = ", sound_data.shape)

sound_data = sound_data / (2.**8)

if len(sound_data.shape) == 1:
    s1 = sound_data
else:
    s1 = sound_data[:,0]

timeArray = np.arange(0, s1.shape[0], 1.0)
timeArray = timeArray / sampling_freq
timeArray = timeArray * 1000

plt.figure(figsize=(20,10))
plt.rcParams.update({'font.size': 25})
plt.title('Plot of sound pressure values over time')
plt.xlabel('Time in milliseconds')
plt.ylabel('Amplitude')
plt.plot(timeArray, sound_data, color='b')
plt.show()