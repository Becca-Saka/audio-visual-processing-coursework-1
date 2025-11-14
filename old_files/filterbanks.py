#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 18:12:48 2025

@author: becca
"""

#
from sound import SoundClass
from soundmag import  magAndPhase
from sound import  plotWaveform,spectrogram
import matplotlib.pyplot as plt
import numpy as np
#
fs = 16000
seconds = 5
sc = SoundClass()
# sc.record(fs, seconds, filename="samp", device=input_device)
r, fs, d = sc.loadfromfile("samp", device=1)
# plotWaveform(r, fs)
frame_start = 22000
frame_length = 512
#

y = r[frame_start : frame_start + frame_length]
magSpec, phaseSpec = magAndPhase(y)
magSpec_half = magSpec[:256]
#
# # ---- Plot it ----
# plt.figure()
# plt.plot(magSpec_half)
# plt.title("Magnitude Spectrum of 512-sample Speech Frame")
# plt.xlabel("Frequency bin (0–255)")
# plt.ylabel("Magnitude")
# plt.show()


def linearRectangularFilterbank(magspec, numChannels):
    N = len(magspec)
    step = N // numChannels
    fbank = np.zeros(numChannels)
    for i in range(numChannels):
        start = i * step
        end = start + step
        fbank[i] = sum(magspec[start:end])
    return fbank

fbank = linearRectangularFilterbank(magSpec, 12)
np.save("bank", fbank)

fig, axs = plt.subplots(2)
axs[0].plot(magSpec)
axs[1].plot(fbank)