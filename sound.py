#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 02:34:25 2025

@author: becca
"""
import sounddevice as sd
import numpy as np
import time
import plotly.io as pio
import plotly.express as px
import pandas as pd
import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
# import numpy as np
pio.renderers.default='browser'
fs = 44100
seconds = 5


class SoundClass():
    """
    Class to record and play sound
    """

    def get_device(self):
        deviceList = sd.query_devices()
        print(deviceList)
        # input_device = 0
        # output_device = 1

    def record(self, fs, seconds, filename="speech", device=None):
        sd.default.device = [3, 2]   # [input, output]
        print("Recording...")
        r = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='float64', device=device)
        sd.wait()   
        print("Playing...")
        sd.play(r, fs)
        sd.wait()
        print("Saving recording...")
        sf.write(f"{filename}.wav", r, fs)

    def record_nsa(self, fs, seconds, device=None):
            sd.default.device = [3, 2]   # [input, output]
            print("Recording...")
            r = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='float64', device=device)
            sd.wait()
            return  r, fs

    def record_ns(self, fs=44100, seconds=10, silence_ms=500, min_talk_ms=200, factor=3.5):
        """
        Record until user stops talking for 'silence_ms' milliseconds.
        Dynamically adjusts to background noise.
        factor: how much above noise RMS to count as speech (3.0–5.0 works well)
        """
        print("Calibrating background noise... stay quiet.")
        frame_len = int(0.03 * fs)  # 30 ms
        silence_limit = int((silence_ms / 1000) / 0.03)
        min_talk_frames = int((min_talk_ms / 1000) / 0.03)

        # --- Step 1: Calibrate baseline noise level ---
        with sd.InputStream(samplerate=fs, channels=1, dtype='float32') as stream:
            noise_frames = []
            for _ in range(10):  # ~0.3s calibration
                frame, _ = stream.read(frame_len)
                noise_frames.append(np.sqrt(np.mean(frame ** 2)))
            baseline = np.mean(noise_frames)
        threshold = baseline * factor
        print(f"✅ Baseline RMS: {baseline:.6f}, speech threshold: {threshold:.6f}")
        print("Recording... speak now 👇")

        buffer = []
        silence_count = 0
        started_talking = False
        start_time = time.time()

        # --- Step 2: Actual recording ---
        with sd.InputStream(samplerate=fs, channels=1, dtype='float32') as stream:
            while True:
                frame, _ = stream.read(frame_len)
                rms = np.sqrt(np.mean(frame ** 2))
                buffer.append(frame)

                if rms > threshold:
                    # print("Started talking.")
                    started_talking = True
                    silence_count = 0
                elif started_talking:
                    silence_count += 1

                # stop if long silence after talking
                if started_talking and silence_count > silence_limit:
                    print("Detected silence — stopping recording.")
                    break
                # print("Silence count: ", silence_count)
                # fallback timeout
                if (time.time() - start_time) > seconds:
                    print("Max recording time reached.")
                    break

        audio = np.concatenate(buffer, axis=0)
        print(f"Recorded duration: {len(audio) / fs:.2f}s")
        return audio, fs

    def play_audio(self, r, fs, device=None):
        sd.play(r, fs, device=device)
        sd.wait()

    def loadfromfile(self, filename="speech", device=None):
        print("Reading from recording...")
        r_in, fs_in = sf.read(f'{filename}.wav', dtype='float32')
        duration = len(r_in) / fs_in
        print(f"Sample number: {fs_in}")
        print(f"r_in: {r_in}")
        print(f"Duration {duration}")
        # self.play_audio(r_in, fs_in, device=device)

        return r_in, fs_in, duration

        


def plotWaveform(r_in, fs_in):
    
    fig = px.line(r_in)
    fig.show()
    t = np.arange(1/fs_in, 1/fs_in + len(r_in)/fs_in, 1/fs_in)
    # t[0:9]

    df = pd.DataFrame({'x': t, 'y': r_in.squeeze()})
    fig = px.line(df, x='x', y='y', labels=dict(x="Time (Seconds)", y="Amplitude"))
    fig.show()

def spectrogram(r, fs):
    # plt.specgram(r, window=np.hamming(512), noverlap=400, NFFT=512, Fs=fs)
    plt.specgram(r, window=np.hamming(64), noverlap=40, NFFT=64, pad_to=512, Fs=fs)
    

# r_in, fs_in = loadfromfile()
# plotWaveform(r_in,fs_in)
# spectrogram(r_in, fs_in)
# SoundClass().record(fs, seconds, "vowel")

# check_data_uniqueness()
