import glob
import pickle
from math import floor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from keras.layers import InputLayer
from keras.models import Sequential
from keras.src.layers import Dense
from keras.src.optimizers import Adam
from keras.utils import to_categorical
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

from sound import SoundClass
from  plot_bar import *
model_name = "linear/model_weights.weights.h5"
encoder_name = "linear/label_encoder.pkl"
normalizer_name = "linear/norm_stats.npz"
output_name = "linear/correct_predictions.png"


def magAndPhase(speechFrame):
    window = np.hamming(len(speechFrame))
    windowedFrame = speechFrame * window
    xF = np.fft.fft(windowedFrame.squeeze())
    magSpec = np.abs(xF)
    phaseSpec = np.angle(xF)

    half = len(magSpec) // 2
    return magSpec[:half], phaseSpec[:half]


def linearRectangularFilterbank(magspec, numChannels):
    step = len(magspec) // numChannels
    fbank = np.zeros(numChannels)
    for i in range(numChannels):
        start = i * step
        end = start + step
        fbank[i] = np.log(np.sum(magspec[start:end]) + 1e-8)

    return fbank


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
    model.add(InputLayer(shape=(32,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(numClasses, activation='softmax'))

    return model


def feature_extraction(r_in, fs_in, has_noise, max_frames=8, ):
    if r_in.ndim == 2:
        r_in = r_in.mean(axis=1)

    if has_noise and random.random() < 0.5:
        r_in = add_noise(r_in, noise_factor=random.uniform(0.002, 0.01))

    duration = 20
    frame_length = int(duration / 1000 * fs_in)
    numFrames = floor(len(r_in) / frame_length)
    all_frame_features = []
    data = []

    for frame in range(numFrames):
        start = frame * frame_length
        end = start + frame_length
        short_frame = r_in[start:end]
        magSpec, phaseSpec = magAndPhase(short_frame)

        fbank = linearRectangularFilterbank(magSpec, 16)

        frames = fbank.shape[0]

        all_frame_features.append(fbank)

    if len(all_frame_features) > 0:
        F = np.vstack(all_frame_features)
        mean_features = F.mean(axis=0)
        std_features = F.std(axis=0)
        data.append(np.concatenate([mean_features, std_features]))

    return data


import random


def add_noise(audio, noise_factor=0.005):
    noise = np.random.randn(len(audio))
    augmented = audio + noise_factor * noise
    return np.clip(augmented, -1.0, 1.0)


def train_model():
    data = []
    labels = []
    max_frames = 8

    for audio_file in sorted(glob.glob('names_audio_wav/*.wav')):
        print("processing ", audio_file)
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
    print(f"Data shape: {len(data)} samples, Labels: {len(labels)} samples")

    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0) + 1e-8
    data = (data - mean) / std
    np.savez(normalizer_name, mean=mean, std=std)

    print("normalised data", data.shape)

    labels, LE = labelEncoder(labels)

    print(data[0])
    print("============")
    print("Using learning rate 0.001")

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
    for i in range(5):
        idx = np.random.randint(0, len(X_test))
        pred = model.predict(np.expand_dims(X_test[idx], axis=0), verbose=0)
        pred_name = LE.inverse_transform([np.argmax(pred)])[0]
        true_name = LE.inverse_transform([np.argmax(y_test[idx])])[0]
        print(f"Actual: {true_name} → Predicted: {pred_name} ({np.max(pred) * 100:.1f}%)")


def test_model(r_in, fs_in):
    test_data = feature_extraction(r_in, fs_in, False)
    print("fs_in", fs_in)

    print("=====")
    model = createModel()
    with open(encoder_name, 'rb') as f:
        LE = pickle.load(f)
    model.load_weights(model_name)
    model.compile(loss='categorical_crossentropy',
                  metrics=['accuracy'], optimizer=Adam(learning_rate=0.001))

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


train_model()

results = []

for audio_file in sorted(glob.glob('names_audio_wav/*.wav')):
    print(f"\n🎤 File: {audio_file}")
    r_in, fs_in = sf.read(audio_file, dtype='float32')
    predicted_name, confidence = test_model(r_in, fs_in)

    file_name = audio_file.split('/')[-1]
    actual_name = ''.join([c for c in file_name.split('.')[0] if not c.isdigit()])

    results.append({
        'Actual Name': actual_name,
        'Predicted Name': predicted_name
    })
plot_bar_chart(output_name, results)

# test_audio_path = 'names_audio_wav/Zack001.wav'
# print(f"\n🎤 File: {test_audio_path}")
# r_in, fs_in = sf.read(test_audio_path, dtype='float32')
# test_model(r_in, fs_in)
# r_in, fs_in = SoundClass().record_ns(fs=44100,seconds= 3)
# test_model(r_in, fs_in)
