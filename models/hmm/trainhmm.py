import sounddevice as sd
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import librosa
import hmmlearn.hmm as hmm
from math import exp
import pickle


def record_sound(filename, duration=1, fs=44100, play=False):
    sd.play( np.sin( 2*np.pi*940*np.arange(fs)/fs )  , samplerate=fs, blocking=True)
    sd.play( np.zeros( int(fs*0.2) ), samplerate=fs, blocking=True)
    data = sd.rec(frames=duration*fs, samplerate=fs, channels=1, blocking=True)
    if play:
        sd.play(data, samplerate=fs, blocking=True)
    sf.write(filename, data=data, samplerate=fs)

def record_data(prefix, n=40, duration=1):
    for i in range(n):
        print('{}_{}.wav'.format(prefix, i))
        record_sound('{}_{}.wav'.format(prefix, i), duration=duration)
        if i % 5 == 4:
            input("Press Enter to continue...")


record_data("len")
record_data("xuong")
record_data('run')

def get_mfcc(filename):
    data, fs = librosa.load(filename, sr=None)
    mfcc = librosa.feature.mfcc(data, sr=fs, n_fft=1024, hop_length=128)
    return mfcc.T


n_sample = 40
data_len = [get_mfcc('len_{}.wav'.format(i)) for i in range(n_sample)]
data_xuong = [get_mfcc('xuong_{}.wav'.format(i)) for i in range(n_sample)]
data_run = [get_mfcc('run_{}.wav'.format(i)) for i in range(n_sample)]


model_len = hmm.GaussianHMM(n_components=30, verbose=True, n_iter=300)
model_len.fit(X=np.vstack(data_len), lengths=[x.shape[0] for x in data_len])

model_xuong = hmm.GaussianHMM(n_components=30, verbose=True, n_iter=300)
model_xuong.fit(X=np.vstack(data_xuong), lengths=[x.shape[0] for x in data_xuong])

model_run = hmm.GaussianHMM(n_components=30, verbose=True, n_iter=300)
model_run.fit(X=np.vstack(data_run), lengths=[x.shape[0] for x in data_run])


with open("model_len.pkl", "wb") as file1: 
    pickle.dump(model_len, file1)

with open("model_xuong.pkl", "wb") as file2:
    pickle.dump(model_xuong, file2)

with open("model_run.pkl", "wb") as file3:
    pickle.dump(model_run, file3)