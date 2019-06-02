import webrtcvad
import collections
import sys
import signal
import pyaudio
import librosa

from array import array
from struct import pack
import wave
import time
from math import exp

from detector import Detector

class SpeechRecognition:
    def __init__(self, file_path, detector):
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.CHUNK_DURATION_MS = 30       # supports 10, 20 and 30 (ms)
        self.PADDING_DURATION_MS = 1500   # 1 sec jugement
        self.CHUNK_SIZE = int(self.RATE * self.CHUNK_DURATION_MS / 1000)  # chunk to read
        self.CHUNK_BYTES = self.CHUNK_SIZE * 2  # 16bit = 2 bytes, PCM
        self.NUM_PADDING_CHUNKS = int(self.PADDING_DURATION_MS / self.CHUNK_DURATION_MS)
        # NUM_WINDOW_CHUNKS = int(240 / CHUNK_DURATION_MS)
        self.NUM_WINDOW_CHUNKS = int(400 / self.CHUNK_DURATION_MS)  # 400 ms/ 30ms  ge
        self.NUM_WINDOW_CHUNKS_END = self.NUM_WINDOW_CHUNKS * 2

        self.START_OFFSET = int(self.NUM_WINDOW_CHUNKS * self.CHUNK_DURATION_MS * 0.5 * self.RATE)

        self.vad = webrtcvad.Vad(1)

        self.stream = pyaudio.PyAudio().open(
            format = self.FORMAT,
            channels = self.CHANNELS,
            rate = self.RATE,
            input = True,
            start = False,
            frames_per_buffer = self.CHUNK_SIZE
        )        

        self.got_a_sentence = False
        self.leave = False
        self.file_path = file_path

        self.LEN = 'len'
        self.XUONG = 'xuong'
        self.RUN = '...'

        # deque of actions from speech
        self.actions = collections.deque(maxlen=100)
        self.detector = detector


    def record_to_file(self, data, sample_width):
        "Records from the microphone and outputs the resulting data to 'path'"
        # sample_width, data = record()
        data = pack('<' + ('h' * len(data)), *data)
        wf = wave.open(self.file_path, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(sample_width)
        wf.setframerate(self.RATE)
        wf.writeframes(data)
        wf.close()


    def normalize(self, snd_data):
        "Average the volume out"
        MAXIMUM = 32767  # 16384
        # MAXIMUM = 30000
        times = float(MAXIMUM) / max(abs(i) for i in snd_data)
        r = array('h')
        for i in snd_data:
            r.append(int(i * times))
        return r


    # detect if audio is speech
    def detect(self):
        ring_buffer = collections.deque(maxlen=self.NUM_PADDING_CHUNKS)
        triggered = False
        ring_buffer_flags = [0] * self.NUM_WINDOW_CHUNKS
        ring_buffer_index = 0

        ring_buffer_flags_end = [0] * self.NUM_WINDOW_CHUNKS_END
        ring_buffer_index_end = 0
        # WangS
        raw_data = array('h')
        index = 0
        start_point = 0
        StartTime = time.time()
        print("* recording: ")
        self.stream.start_stream()

        while not self.got_a_sentence and not self.leave:
            chunk = self.stream.read(self.CHUNK_SIZE)
            # add WangS
            raw_data.extend(array('h', chunk))
            index += self.CHUNK_SIZE
            TimeUse = time.time() - StartTime

            active = self.vad.is_speech(chunk, self.RATE)

            sys.stdout.write('1' if active else '_')
            ring_buffer_flags[ring_buffer_index] = 1 if active else 0
            ring_buffer_index += 1
            ring_buffer_index %= self.NUM_WINDOW_CHUNKS

            ring_buffer_flags_end[ring_buffer_index_end] = 1 if active else 0
            ring_buffer_index_end += 1
            ring_buffer_index_end %= self.NUM_WINDOW_CHUNKS_END

            # start point detection
            if not triggered:
                ring_buffer.append(chunk)
                num_voiced = sum(ring_buffer_flags)
                if num_voiced > 0.8 * self.NUM_WINDOW_CHUNKS:
                    sys.stdout.write(' Open ')
                    triggered = True
                    start_point = index - self.CHUNK_SIZE * 20  # start point
                    ring_buffer.clear()
            # end point detection
            else:
                ring_buffer.append(chunk)
                num_unvoiced = self.NUM_WINDOW_CHUNKS_END - sum(ring_buffer_flags_end)
                if num_unvoiced > 0.90 * self.NUM_WINDOW_CHUNKS_END or TimeUse > 10:
                    sys.stdout.write(' Close ')
                    triggered = False
                    self.got_a_sentence = True

            sys.stdout.flush()

        sys.stdout.write('\n')
        # data = b''.join(voiced_frames)

        self.stream.stop_stream()
        print("* done recording")
        self.got_a_sentence = False

        # write to file
        raw_data.reverse()
        for index in range(start_point):
            raw_data.pop()
        raw_data.reverse()
        # raw_data = self.normalize(raw_data)
        self.record_to_file(raw_data, 2)
        # self.leave = True
        
        action = self.detector.predict(self.file_path)
        # if action == self.LEN or action == self.XUONG:
        #     self.actions.append(action)
        return action

    # start streaming audio
    # def start(self):
    #     while True:
    #         if self.leave:
    #             break
    #         self.detect()

    # stop streaming audio
    def stop(self):
        self.leave = True

    # remove the action in the left of deque
    def pop(self):
        if len(self.actions) > 0:
            return self.actions.popleft()
        else:
            return self.RUN

    def size(self):
        return len(self.actions)
