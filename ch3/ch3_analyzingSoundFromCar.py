from scipy.io import wavfile

import numpy as np

import matplotlib.pyplot as plt

AUDIO_FILE = "../data/sound_sample_car_engine.wav"

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



# number of points for fft
n = len(s1)
# take the Fourier transform
p = np.fft.fft(s1)

# only half the points will give us the frequency bins
nUniquePts = int(np.ceil((n+1)/2.0))
p = p[0:nUniquePts]
p=abs(p)

# create the array of freqneyc points

freqArray = np.arange(0,float(nUniquePts), 1.0) * float(sampling_freq) / n;

#convert the frequency from hertz to engine RPM
MAX_RPM = 20000
NUM_POINTS = 20

#remove points above max RPM
maxhz = MAX_RPM / 60
p[freqArray > maxhz] = 0

# plot the frequency domain plot
plt.figure(figsize=(20,10))
plt.rcParams.update({'font.size': 25})
plt.title('Plot of sound waves in frequency domain')
plt.plot(freqArray*60, p, color='r')
plt.xlabel('Engine RPM')
plt.ylabel('Signal Power (dB)')
plt.xlim([0,MAX_RPM])
plt.xticks(np.arange(0, MAX_RPM, MAX_RPM/NUM_POINTS),
size='small',rotation=40)
plt.grid()
plt.show()


