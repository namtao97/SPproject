from detector import Detector
from recorder import SpeechRecognition
from threading import Thread


detector = Detector(model_len_path='pyaudio/model_len.pkl', model_xuong_path='pyaudio/model_xuong.pkl')
sp = SpeechRecognition('recording.wav', detector=detector)

while True:
    action = sp.detect()
    print(action)

# action = detector.predict('recording.wav')
# print(action)