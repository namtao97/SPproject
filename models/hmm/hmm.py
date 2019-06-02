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

#record_sound('test.wav')

def get_mfcc(filename):
    data, fs = librosa.load(filename, sr=None)
    mfcc = librosa.feature.mfcc(data, sr=fs, n_fft=1024, hop_length=128)
    return mfcc.T

# n_sample = 25
# data_len = [get_mfcc('len_{}.wav'.format(i)) for i in range(n_sample)]
# data_xuong = [get_mfcc('xuong_{}.wav'.format(i)) for i in range(n_sample)]


# model_len = hmm.GaussianHMM(n_components=30, verbose=True, n_iter=200)
# model_len.fit(X=np.vstack(data_len), lengths=[x.shape[0] for x in data_len])

# model_xuong = hmm.GaussianHMM(n_components=30, verbose=True, n_iter=200)
# model_xuong.fit(X=np.vstack(data_xuong), lengths=[x.shape[0] for x in data_xuong])

# mfcc = get_mfcc('xuong_0.wav')
# model_len.score(mfcc), model_xuong.score(mfcc)

with open("model_len.pkl", "rb") as file1:
    model_len = pickle.load(file1)

with open("model_xuong.pkl", "rb") as file2:
    model_xuong = pickle.load(file2)

with open('model_run.pkl', 'rb') as file3:
    model_run = pickle.load(file3)

def get_prob(log_x1, log_x2):
    if log_x1 < log_x2:
        exp_x1_x2 = exp(log_x1-log_x2)
        return exp_x1_x2 / (1+exp_x1_x2), 1 / (1+exp_x1_x2)
    else:
        p = get_prob(log_x2, log_x1)
        return p[1], p[0]

log_len_threshole = -24000

while True:
    record_sound('nam.wav')
    mfcc = get_mfcc('nam.wav')
    log_plen, log_pxuong, log_prun = model_len.score(mfcc), model_xuong.score(mfcc), model_run.score(mfcc)
    print(log_plen, log_pxuong)

    # if log_plen == min([log_plen, log_pxuong, log_prun]):
    #     print('len')
    # elif log_pxuong == min([log_plen, log_pxuong, log_prun]):
    #     print('xuong')
    # else:
    #     print('...')

    plen, pxuong = get_prob(log_plen, log_pxuong)
    if plen > pxuong:
        if log_plen < log_len_threshole:
            print('len')
        else:
            print('...')
    else:
        print("xuong")
    # print(plen, pxuong, "len" if plen > pxuong else "xuong")