import pickle
import librosa
from math import exp

class Detector:
    def __init__(self, model_len_path='hmm/model_len.pkl', model_xuong_path='hmm/model_xuong.pkl'):
        self.LEN = 'len'
        self.XUONG = 'xuong'
        self.RUN = '...'

        self.log_len_threshole = -24000
        self.log_xuong_threshole = -24000

        self.model_len, self.model_xuong = self.load_model(model_len_path, model_xuong_path)
    
    # load pre-trained models
    def load_model(self, model_len_path, model_xuong_path):
        with open(model_len_path, "rb") as file1:
            model_len = pickle.load(file1)
        with open(model_xuong_path, "rb") as file2:
            model_xuong = pickle.load(file2)

        return model_len, model_xuong


    def get_mfcc(self, file_path):
        data, fs = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(data, sr=fs, n_fft=1024, hop_length=128)
        return mfcc.T


    def get_prob(self, log_x1, log_x2):
        if log_x1 < log_x2:
            exp_x1_x2 = exp(log_x1-log_x2)
            return exp_x1_x2 / (1+exp_x1_x2), 1 / (1+exp_x1_x2)
        else:
            p = self.get_prob(log_x2, log_x1)
            return p[1], p[0]


    def predict(self, file_path):
        mfcc = self.get_mfcc(file_path)

        # prediction
        log_plen, log_pxuong = self.model_len.score(mfcc), self.model_xuong.score(mfcc)
        # print(log_plen, log_pxuong)

        plen, pxuong = self.get_prob(log_plen, log_pxuong)
        if plen > pxuong:
            if log_plen > self.log_len_threshole:
                # print(self.LEN)
                return self.LEN
            else:
                # print(self.RUN)
                return self.RUN
        else:
            if log_pxuong > self.log_xuong_threshole:
                # print(self.XUONG)
                return self.XUONG
            else:
                # print(self.RUN)
                return self.RUN


