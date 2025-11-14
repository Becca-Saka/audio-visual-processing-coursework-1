import glob
import pickle
import random
import numpy as np
import soundfile as sf
from keras.layers import InputLayer
from keras.models import Sequential
from keras.src.layers import Dense
from keras.src.optimizers import Adam
from keras.utils import to_categorical
from scipy.fftpack import dct
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from plot_bar import *
from sound import SoundClass

model_name = "mel/model_weights.weights.h5"
encoder_name = "mel/label_encoder.pkl"
normalizer_name = "mel/norm_stats.npz"
output_name = "mel/correct_predictions.png"

def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700.0)


def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595.0) - 1)


def mel_filterbank(num_filters=26, n_fft=512, fs=16000, min_hz=0, max_hz=None):
    if max_hz is None:
        max_hz = fs / 2
    min_mel = hz_to_mel(min_hz)
    max_mel = hz_to_mel(max_hz)
    mel_points = np.linspace(min_mel, max_mel, num_filters + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / fs).astype(int)

    fbanks = np.zeros((num_filters, n_fft // 2 + 1))
    for m in range(1, num_filters + 1):
        f_m_minus = bins[m - 1]
        f_m = bins[m]
        f_m_plus = bins[m + 1]
        for k in range(f_m_minus, f_m):
            fbanks[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            fbanks[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)
    return fbanks


def labelEncoder(labels):
    LE = LabelEncoder()

    classes = ['Ahmed', 'Amber', 'Charlie', 'Christopher', 'Dominic', 'Emad', 'Emma', 'Hannah', 'Imogen', 'Jess',
               'Josh', 'Joshua', 'Kailong', 'Kira', 'Manwel', 'Mateusz', 'Ngozi', 'Riley', 'Sivaprasath', 'Zack']
    LE = LE.fit(classes)
    transformed_label = to_categorical(LE.transform(labels))
    return transformed_label, LE


def createModel():
    model = Sequential()

    numClasses = 20
    model.add(InputLayer(shape=(26,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(numClasses, activation='softmax'))

    return model


def feature_extraction(r_in, fs_in, has_noise, max_frames=8):
    if r_in.ndim == 2:
        r_in = r_in.mean(axis=1)

    if has_noise and random.random() < 0.5:
        r_in = add_noise(r_in, noise_factor=random.uniform(0.002, 0.01))

    frame_len = int(0.025 * fs_in)
    frame_step = int(0.010 * fs_in)
    numFrames = 1 + int((len(r_in) - frame_len) / frame_step)
    all_frame_features = []

    mel_fbanks = mel_filterbank(num_filters=26, n_fft=frame_len, fs=fs_in)

    for i in range(numFrames):
        start = i * frame_step
        end = start + frame_len
        if end > len(r_in): break
        short_frame = r_in[start:end] * np.hamming(frame_len)

        magSpec = np.abs(np.fft.rfft(short_frame))
        mel_energy = np.dot(mel_fbanks, magSpec)
        fbank = np.log(mel_energy + 1e-8)

        mfcc = dct(fbank, type=2, norm='ortho')[:13]
        all_frame_features.append(mfcc)

    if len(all_frame_features) > 0:
        F = np.vstack(all_frame_features)
        mean_features = F.mean(axis=0)
        std_features = F.std(axis=0)
        return [np.concatenate([mean_features, std_features])]
    else:
        return []


def add_noise(audio, noise_factor=0.005):
    noise = np.random.randn(len(audio))
    augmented = audio + noise_factor * noise
    return np.clip(augmented, -1.0, 1.0)


def train_model():
    data = []
    labels = []
    max_frames = 8

    for audio_file in sorted(glob.glob('names_audio_wav/*.wav')):

        r_in, fs_in = sf.read(f'{audio_file}', dtype='float32')
        feats = feature_extraction(r_in, fs_in, True, max_frames)
        if len(feats) > 0:
            data += feats
            label = audio_file.split('/')[-1].split('.')[0]
            label = label[0:-3]
            labels.append(label)

        continue

    data = np.array(data)
    labels = np.array(labels)

    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0) + 1e-8
    data = (data - mean) / std
    np.savez(normalizer_name, mean=mean, std=std)

    labels, LE = labelEncoder(labels)

    data, labels = shuffle(data, labels, random_state=0)

    X_train, X_tmp, y_train, y_tmp = train_test_split(data, labels, test_size=0.2, random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=0)

    model = createModel()

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(learning_rate=0.001))
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32, epochs=60, verbose=1)
    model.summary()
    model.save_weights(model_name)
    with open(encoder_name, 'wb') as f:
        pickle.dump(LE, f)
    predicted_probabilities = model.predict(X_test, verbose=0)
    predicted = np.argmax(predicted_probabilities, axis=1)
    actual = np.argmax(y_test, axis=1)
    accuracy = metrics.accuracy_score(actual, predicted)
    print("accuracy:", accuracy * 100)

    predicted_prob = model.predict(np.expand_dims(X_test[0, :], axis=0), verbose=1)

    predicted_id = np.argmax(predicted_prob, axis=1)
    predicted_class = LE.inverse_transform(predicted_id)
    print("predicted class:", predicted_class)

    confusion_matrix = metrics.confusion_matrix(np.argmax(y_test, axis=1), predicted)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    cm_display.plot()
    return model, LE


def test_model(r_in, fs_in, model, LE):
    test_data = feature_extraction(r_in, fs_in, False)

    # model = createModel()
    # with open(encoder_name, 'rb') as f:
    #     LE = pickle.load(f)
    # model.load_weights(model_name)
    # model.compile(loss='categorical_crossentropy',
    #               metrics=['accuracy'], optimizer=Adam(learning_rate=0.001))

    stats = np.load(normalizer_name)
    mean, std = stats['mean'], stats['std']
    test_data = (test_data - mean) / std

    pred = model.predict(test_data, verbose=0)

    predicted_id = np.argmax(pred, axis=1)
    predicted_name = LE.inverse_transform(predicted_id)[0]
    confidence = f"{np.max(pred) * 100:.2f}%"
    print(f"Predicted Name: {predicted_name}")
    print(f"Confidence: {confidence}")
    return predicted_name, confidence
model, LE = train_model()
results = []
for audio_file in sorted(glob.glob('test_audios/*.wav')):
    r_in, fs_in = sf.read(audio_file, dtype='float32')
    # print(f"Testing with audio, {audio_file} and sample rate {fs_in}")
    if r_in.ndim == 2:
        r_in = r_in.mean(axis=1)
    predicted_name, confidence = test_model(r_in, fs_in, model, LE)
    audio_file = audio_file.split('/')[-1]
    actual_name = ''.join([c for c in audio_file.split('.')[0] if not c.isdigit()])

    results.append({
            'Actual Name': actual_name,
            'Predicted Name': predicted_name.strip(),
        })


import pprint
pprint.pprint(results)

#
# results = []

# for audio_file in sorted(glob.glob('names_audio_wav/*.wav')):
#     r_in, fs_in = sf.read(audio_file, dtype='float32')
#     predicted_name, confidence = test_model(r_in, fs_in)
#
#     file_name = audio_file.split('/')[-1]
#     actual_name = ''.join([c for c in file_name.split('.')[0] if not c.isdigit()])
#
#     results.append({
#         'Actual Name': actual_name,
#         'Predicted Name': predicted_name
#     })
#
# plot_bar_chart(output_name, results)

# test_audio_path = 'names_audio_wav/Zack001.wav'
# print(f"\n🎤 File: {test_audio_path}")
# r_in, fs_in = sf.read(test_audio_path, dtype='float32')
# test_model(r_in, fs_in)
# r_in, fs_in = SoundClass().record_ns(fs=44100,seconds= 3)
# test_model(r_in, fs_in)
