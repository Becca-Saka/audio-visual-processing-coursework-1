from math import floor

from keras.src.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import InputLayer
from keras.src.layers import Dense
import soundfile as sf
import numpy as np
import glob


# Feature Extraction
def magAndPhase(speechFrame):
    window = np.hamming(len(speechFrame))
    windowedFrame = speechFrame * window
    xF = np.fft.fft(windowedFrame.squeeze())
    magSpec = np.abs(xF)
    phaseSpec = np.angle(xF)
    return magSpec, phaseSpec


def linearRectangularFilterbank(frame_length, magspec, numChannels):
    # FIXED: Use spectrum length, not frame_length
    spec_len = len(magspec) // 2  # Only use positive frequencies
    step = spec_len // numChannels

    fbank = np.zeros(numChannels)
    for i in range(numChannels):
        start = i * step
        end = start + step
        if end > spec_len:
            end = spec_len
        # FIXED: Use mean instead of sum
        fbank[i] = np.mean(magspec[start:end])

    # FIXED: Apply log compression
    fbank = np.log(fbank + 1e-8)
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
    # FIXED: Changed to 13 features and increased network size
    model.add(InputLayer(input_shape=(13,)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(numClasses, activation='softmax'))
    return model


from sklearn import metrics
from keras.layers import Dropout
from sklearn.preprocessing import StandardScaler


def compute_deltas(features):
    """Compute delta (first derivative) of features"""
    deltas = np.zeros_like(features)
    for i in range(1, len(features) - 1):
        deltas[i] = (features[i + 1] - features[i - 1]) / 2
    return deltas


data = []
labels = []

for audio_file in sorted(glob.glob('names_audio_wav/*.wav')):
    r_in, fs_in = sf.read(f'{audio_file}', dtype='float32')

    if r_in.ndim > 1:
        r_in = r_in[:, 0]

    duration = 20
    frame_length = int(duration / 1000 * fs_in)
    numFrames = floor(len(r_in) / frame_length)

    all_frame_features = []

    for frame in range(numFrames):
        start = frame * frame_length
        end = start + frame_length
        short_frame = r_in[start:end]
        magSpec, phaseSpec = magAndPhase(short_frame)
        fbank = linearRectangularFilterbank(frame_length, magSpec, 20)
        all_frame_features.append(fbank)

    if len(all_frame_features) > 0:
        all_frame_features = np.array(all_frame_features)

        # Static features
        mean_features = np.mean(all_frame_features, axis=0)
        std_features = np.std(all_frame_features, axis=0)

        # Delta features
        delta_features = compute_deltas(all_frame_features)
        delta_mean = np.mean(delta_features, axis=0)

        # Combine
        combined_features = np.concatenate([mean_features, std_features, delta_mean])
        data.append(combined_features)

        label = audio_file.split('/')[-1].split('.')[0]
        label = label[0:-3]
        labels.append(label)

data = np.array(data)
labels = np.array(labels)

# Better normalization
scaler = StandardScaler()
data = scaler.fit_transform(data)

print("normalized data", data.shape)

labels, LE = labelEncoder(labels)

X_train, X_tmp, y_train, y_tmp = train_test_split(data, labels, test_size=0.2, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=0)


def createModel():
    model = Sequential()
    numClasses = 20
    model.add(InputLayer(input_shape=(60,)))  # 20 features * 3
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(numClasses, activation='softmax'))
    return model


model = createModel()
model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
              optimizer=Adam(learning_rate=0.001))
history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    batch_size=16, epochs=100, verbose=1)
model.summary()
model.save_weights('model_weights.weights.h5')

predicted_probabilities = model.predict(X_test, verbose=0)
predicted = np.argmax(predicted_probabilities, axis=1)
actual = np.argmax(y_test, axis=1)
accuracy = metrics.accuracy_score(actual, predicted)
print("accuracy:", accuracy * 100)

predicted_prob = model.predict(np.expand_dims(X_test[0, :], axis=0), verbose=1)
print("predicted_prob:", predicted_prob)
predicted_id = np.argmax(predicted_prob, axis=1)
predicted_class = LE.inverse_transform(predicted_id)
print("predicted class:", predicted_class)

confusion_matrix = metrics.confusion_matrix(np.argmax(y_test, axis=1), predicted)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
cm_display.plot()