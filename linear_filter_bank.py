import glob
from math import floor
import numpy as np
import soundfile as sf
import random
import math

max_frames = 83

class FilterBank:

    def mag_and_phase(self, speech_frame):
        window = np.hamming(len(speech_frame))
        windowed_frame = speech_frame * window
        xF = np.fft.fft(windowed_frame.squeeze())
        mag_spec = np.abs(xF)
        phase_spec = np.angle(xF)
        half = len(mag_spec) // 2
        return mag_spec[:half], phase_spec[:half]

    def linear_rectangular_filterbank(self, magspec, num_channels):
        step = len(magspec) // num_channels
        fbank = np.zeros(num_channels)
        for i in range(num_channels):
            start = i * step
            end = start + step
            fbank[i] = np.log(np.sum(magspec[start:end]) + 1e-8)

        return fbank

    def feature_extraction(self, r_in, fs_in, has_noise):

        if r_in.ndim == 2:
            r_in = r_in.mean(axis=1)

        if has_noise:
            r_in = self.add_noise(r_in, fs_in)

        duration = 20
        frame_length = int(duration / 1000 * fs_in)
        numFrames = floor(len(r_in) / frame_length)
        all_frame_features = []
        data = []


        for frame in range(numFrames):
            start = frame * frame_length
            end = start + frame_length
            short_frame = r_in[start:end]
            magSpec, phaseSpec = self.mag_and_phase(short_frame)

            fbank = self.linear_rectangular_filterbank(magSpec, 32)

            all_frame_features.append(fbank)

        if len(all_frame_features) > 0:
            F = np.vstack(all_frame_features)
            if F.shape[0] < max_frames:
                pad_width = max_frames - F.shape[0]
                F = np.pad(F, ((0, pad_width), (0, 0)), mode='constant')

            mean_features = F.mean(axis=0)
            std_features = F.std(axis=0)

            data.append(np.concatenate([mean_features, std_features]))

        return data

    def add_noise(self, audio, fs_in):
        def signalPower(s):
            p = np.mean(s ** 2)
            return p

        def amplitude(sp, nsp, snr):
            a = math.sqrt((sp / nsp) * (10 ** (-snr / 10)))
            return a

        noise_files = ['lab_noise.wav', 'library_noise.wav']
        random_index = random.randint(0, len(noise_files) - 1)
        noise_file = noise_files[random_index]
        nr_in, nfs_in = sf.read(noise_file, dtype='float32')

        sp_x = signalPower(audio)
        sp_d = signalPower(nr_in)
        a = amplitude(sp_x, sp_d, 0)

        if len(nr_in) < len(audio):
            # Repeats noise to cover full audio length
            repeats = int(np.ceil(len(audio) / len(nr_in)))
            nr_in = np.tile(nr_in, repeats)[:len(audio)]
        elif len(nr_in) > len(audio):
            # Trims noise to same length as audio
            nr_in = nr_in[:len(audio)]
        y = audio + a * nr_in

        return y

    def get_training_features(self):
        data = []
        labels = []

        for audio_file in sorted(glob.glob('src/names_audio_wav/*.wav')):
            r_in, fs_in = sf.read(f'{audio_file}', dtype='float32')
            # print(f"processing with sample rate {fs_in} ", audio_file)

            label = audio_file.split('/')[-1].split('.')[0]
            label_number = ''.join([c for c in label if c.isdigit()])
            label_number = int(label_number)

            label = ''.join([c for c in label if not c.isdigit()])


            if 10 < label_number < 20:
                feats = self.feature_extraction(r_in, fs_in, has_noise=False)
            else:
                feats = self.feature_extraction(r_in, fs_in, has_noise=False)

            if len(feats) > 0:
                data += feats
                labels.append(label)

        data = np.array(data)
        labels = np.array(labels)
        return data, labels