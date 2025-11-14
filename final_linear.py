import glob
import pickle
from math import floor
from operator import index
import pprint
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from keras.layers import InputLayer
from keras.models import Sequential
from keras.src.layers import Dense
from keras.src.optimizers import Adam
from keras.utils import to_categorical
from numba.core.cgutils import false_bit
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import random
import math

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
    model.add(InputLayer(shape=(64,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(numClasses, activation='softmax'))

    return model

max_frames = 83

def feature_extraction(r_in, fs_in, has_noise):

    if r_in.ndim == 2:
        r_in = r_in.mean(axis=1)

    if has_noise:
        r_in = add_noise(r_in, fs_in)


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

        fbank = linearRectangularFilterbank(magSpec, 32)

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





def add_noise(audio, fs_in):
    def signalPower(s):
        p = np.mean(s ** 2)
        return p

    def amplitude(sp, nsp, snr):
        a = math.sqrt((sp / nsp) * (10 ** (-snr / 10)))
        return a

    noise_files = ['noise.wav', 'noise_2.wav']
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

max_frames = 83
def train_model():
    data = []
    labels = []


    for audio_file in sorted(glob.glob('names_audio_wav/*.wav')):
        r_in, fs_in = sf.read(f'{audio_file}', dtype='float32')
        # print(f"processing with sample rate {fs_in} ", audio_file)
        # get digits in file name
        label = audio_file.split('/')[-1].split('.')[0]
        label_number = ''.join([c for c in label if c.isdigit()])
        label_number = int(label_number)


        if 10 < label_number < 20:
            feats = feature_extraction(r_in, fs_in, has_noise=False)
        else:
            feats = feature_extraction(r_in, fs_in, has_noise=False)
            # label_number = label_number[:max_frames]
        # print("Digit name", label_number)

        if len(feats) > 0:
            data += feats
            label = ''.join([c for c in label if not c.isdigit()])  # keep only letters
            labels.append(label)





    data = np.array(data)
    labels = np.array(labels)
    # print("Unique labels:", np.unique(labels))
    # from collections import Counter
    # print(Counter(labels))

    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0) + 1e-8
    np.savez(normalizer_name, mean=mean, std=std)
    data = (data - mean) / std
    # print("normalised data", data.shape)

    labels, LE = labelEncoder(labels)

    # print(data[0])


    # data, labels = shuffle(data, labels, random_state=0)

    X_train, X_tmp, y_train, y_tmp = train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)

    model = createModel()

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(learning_rate=0.001))
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32, epochs=60, verbose=1)
    model.summary()
    model.save_weights(model_name)
    # with open(encoder_name, 'wb') as f:
    #     pickle.dump(LE, f)
    predicted_probabilities = model.predict(X_test, verbose=0)
    predicted = np.argmax(predicted_probabilities, axis=1)
    actual = np.argmax(y_test, axis=1)
    accuracy = metrics.accuracy_score(actual, predicted)
    print("accuracy:", accuracy * 100)

    predicted_prob = model.predict(np.expand_dims(X_test[0, :], axis=0), verbose=1)

    predicted_id = np.argmax(predicted_prob, axis=1)
    # predicted_class = LE.inverse_transform(predicted_id)
    # print("predicted class:", predicted_class)

    # from sklearn.metrics import classification_report
    # print("\nPer-class performance:")
    # unique_labels = np.unique(np.concatenate([actual, predicted]))
    # target_names = [LE.classes_[i] for i in unique_labels]
    # print("classification_report", classification_report(actual, predicted, labels=unique_labels, target_names=target_names))

    confusion_matrix = metrics.confusion_matrix(np.argmax(y_test, axis=1), predicted)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    cm_display.plot()
    # plt.show()
    plt.savefig("confusion_matrix.png")
    for i in range(5):
        idx = np.random.randint(0, len(X_test))
        pred = model.predict(np.expand_dims(X_test[idx], axis=0), verbose=0)
        pred_name = LE.inverse_transform([np.argmax(pred)])[0]
        true_name = LE.inverse_transform([np.argmax(y_test[idx])])[0]
        print(f"Actual: {true_name} → Predicted: {pred_name} ({np.max(pred) * 100:.1f}%)")
    print("============")
    return  model, LE


def test_model(r_in, fs_in, model, LE):

    test_data = feature_extraction(r_in, fs_in, False)




    # print("test_data shape before norm:", np.array(test_data).shape)
    #


    stats = np.load(normalizer_name)
    mean, std = stats['mean'], stats['std']
    # print(stats['mean'][:5], stats['std'][:5])
    # print("mean/std shape:", mean.shape, std.shape)


    test_data = np.array(test_data).reshape(-1, 64)  # <-- fix shape here
    test_data = (test_data - mean) / std

    pred = model.predict(test_data, verbose=0)
    top_five = top5(pred, LE)
    for i in top_five:
        print(f"Pred at {i}")

    predicted_id = np.argmax(pred, axis=1)
    # print("predicted class:", predicted_id)
    predicted_name = LE.inverse_transform(predicted_id)[0]

    confidence = f"{np.max(pred) * 100:.2f}%"
    # print(f"Predicted Name: {predicted_name}")
    # print(f"Confidence: {confidence}")
    print("=====")
    return predicted_name, confidence

def top5(pred, LE):
    top_idx = np.argsort(pred[0])[::-1][:5]
    return [(LE.inverse_transform([i])[0], float(pred[0][i])) for i in top_idx]




model, LE = train_model()
r_in, fs_in = SoundClass().record_ns(fs=44100,seconds= 3)
test_model(r_in, fs_in, model, LE)

# results = []
# for audio_file in sorted(glob.glob('test_audios/*.wav')):
#     r_in, fs_in = sf.read(audio_file, dtype='float32')
#     if r_in.ndim == 2:
#         r_in = r_in.mean(axis=1)
#     # for i in range(5):
#     print(f"Testing with audio, {audio_file} and sample rate {fs_in}")
#
#     predicted_name, confidence = test_model(r_in, fs_in, model, LE)
#     audio_file = audio_file.split('/')[-1]
#     actual_name = ''.join([c for c in audio_file.split('.')[0] if not c.isdigit()])
#     results.append({
#             'Actual Name': actual_name,
#             'Predicted Name': predicted_name.strip(),
#         })
#
#
#
# pprint.pprint(results)


# results = []
# for audio_file in sorted(glob.glob('names_audio_wav/*.wav')):
#     print(f"\n🎤 File: {audio_file}")
#     r_in, fs_in = sf.read(audio_file, dtype='float32')
#     predicted_name, confidence = test_model(r_in, fs_in, model, LE)
#
#     file_name = audio_file.split('/')[-1]
#     actual_name = ''.join([c for c in file_name.split('.')[0] if not c.isdigit()])
#
#     results.append({
#         'Actual Name': actual_name,
#         'Predicted Name': predicted_name
#     })
# plot_bar_chart(output_name, results)