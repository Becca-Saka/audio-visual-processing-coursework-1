from math import floor

from keras.src.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import InputLayer
from keras.src.layers import Dense
import soundfile as sf
import  numpy as np

import glob



# Feature Extraction
def magAndPhase(speechFrame):
    window = np.hamming(len(speechFrame))
    # print("SpeechFrame ", speechFrame.shape)
    # print("window ", window.shape)
    windowedFrame = speechFrame * window
    xF = np.fft.fft(windowedFrame.squeeze())
    magSpec = np.abs(xF)
    phaseSpec = np.angle(xF)
    # print("magSpec ", magSpec)
    # print("phaseSpec ", phaseSpec)
    return magSpec, phaseSpec

def linearRectangularFilterbank(frame_length, magspec, numChannels):
    # step = 32
    step = int(frame_length)
    fbank = np.zeros(numChannels)
    for i in range(numChannels):
        start = i * step
        end = start + step
        fbank[i] = sum(magspec[start:end])
    return fbank

def matrixFB(frame_length, magSpec, numChannels):
    step = int(frame_length)
    fbank = np.zeros(numChannels)
    for i in range(numChannels):
        r = np.zeros(512)
        start = i * step
        end = start + step
        r[start:end] = 1
        fbank[i] = np.matmul(r, magSpec[0:512])
    return fbank

def labelEncoder(labels):
    LE = LabelEncoder()


    #create an array of 0-19 string
    # classes = ['Ahmed', 'Amber', 'Charlie', 'Christopher', 'Dominic']
    classes = ['Ahmed', 'Amber', 'Charlie', 'Christopher', 'Dominic', 'Emad', 'Emma', 'Hannah', 'Imogen', 'Jess', 'Josh', 'Joshua', 'Kailong', 'Kira', 'Manwel', 'Mateusz', 'Ngozi', 'Riley', 'Sivaprasath', 'Zack']
    LE = LE.fit(classes)
    transformed_label = to_categorical(LE.transform(labels))
    # print("labels", transformed_label)
    return transformed_label, LE

def createModel():
    model = Sequential()
    # numClasses = 5
    numClasses = 20
    model.add(InputLayer(input_shape=(8,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(numClasses, activation='softmax'))
    # model.add(InputLayer(input_shape=8))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Flatten())
    # model.add(Dense(256))
    # model.add(Activation('relu'))
    # model.add(Dense(numClasses))
    # model.add(Activation('softmax'))
    return model


from sklearn import metrics

data = []
data_copy =[]
labels = []
max_frames =8
counter = 0

# Read Audio

# for  audio_file in sorted(glob.glob('sample_wav/*.wav')):
for  audio_file in sorted(glob.glob('names_audio_wav/*.wav')):
    # print("processing ", audio_file)
    # if counter >= 5:
    #     break
    # counter +=1

    r_in, fs_in = sf.read(f'{audio_file}', dtype='float32')
    # print("number of shape", r_in.shape)
    # print("number of dimensions", r_in.ndim)
    if r_in.ndim > 1:
        r_in = r_in[:,0]

    duration = 20
    # duration = len(r_in) / fs_in
    frame_length = int( duration/1000 * fs_in)
    numFrames = floor(len(r_in) / frame_length)
    # print("numFrames", numFrames)

    for frame in range(numFrames):
        start = frame * frame_length
        end = start + frame_length
        short_frame = r_in[start:end]
        magSpec, phaseSpec = magAndPhase(short_frame)
        # print("length of magSpec", len(magSpec))

        # print("Before filter", r_in.shape)
        fbank = linearRectangularFilterbank(frame_length,magSpec, 8)

        frames = fbank.shape[0]
        # print("Before padding", fbank.shape)
        # if frames < max_frames:
        #     fbank = np.pad(fbank, ((0,0), (0, max_frames-fbank.shape[1])) )
            # print("Afrer padding", fbank.shape)
        # max_frames = max(max_frames, frames)
        data.append(fbank)
        label = audio_file.split('/')[-1].split('.')[0]
        label = label[0:-3]
        labels.append(label)
    # print("=====")



#Convert to numpy for NN
data = np.array(data)
labels = np.array(labels)

#Normalize data
data = data/np.max(data)
print("normalised data" , data.shape)



# np.save('data', data)
# np.save('labels', labels)

labels,LE = labelEncoder(labels)
# print('data shape', data.shape)
# print('label shape', labels.shape, labels[0:5])
print(data[0])
print("============")
X_train,X_tmp, y_train, y_tmp = train_test_split(data, labels, test_size=0.2, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=0)

# integer_labels = np.argmax(y_train, axis=1)
# print(integer_labels)


model = createModel()

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(learning_rate=0.01))
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32,epochs=30,verbose=1)
model.summary()
model.save_weights('model_weights.weights.h5')
predicted_probabilities = model.predict(X_test, verbose=0)
predicted = np.argmax(predicted_probabilities, axis=1)
actual = np.argmax(y_test, axis=1)
accuracy = metrics.accuracy_score(actual, predicted)
print("accuracy:", accuracy *100)



# integer_labels = np.argmax(X_test, axis=0)
# print("integer_labels", integer_labels)

# predicted_prob = model.predict(np.expand_dims(X_test[0], axis=0), verbose=0)

predicted_prob = model.predict(np.expand_dims(X_test[0,:], axis=0), verbose=1)
print("predicted_prob:", predicted_prob)
predicted_id = np.argmax(predicted_prob, axis=1)
predicted_class = LE.inverse_transform(predicted_id)
print("predicted class:", predicted_class)

# plt.imshow(X_test[0,:], cmap='viridis')
# plt.title("Predicted Class")
# plt.colorbar()
# plt.show()

confusion_matrix = metrics.confusion_matrix(np.argmax(y_test, axis=1), predicted)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
cm_display.plot()






