from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
import numpy as np
import cv2
import os
import sys
import re
import pickle
from utilities import ImageUtilities as IU

class NNSorter(object):

    def __init__(self, model='128D'):
        if model == '128D':
            self.neurons = 128
        else:
            self.neurons = 512
        np.random.seed(19)
        self.folder = None
        self.utils = IU()
        self.face_model = 'hog'
        self.encoding_model = '128D'
        self.jitters = 3
        self.upsample = 1

    def set_folder(self, folder):
        self.folder = folder

    def set_params(self, model, encoding, jitters, upsample):
        self.face_model = model
        self.encoding_model = encoding
        self.jitters = jitters
        self.upsample = upsample

    def train(self):
        model = Sequential()
        model.add(Dense(self.neurons+4, input_dim=self.neurons,
                        activation='relu'))
        model.add(Dropout(0.2, noise_shape=None, seed=None))
        model.add(Dense(int((self.neurons*2)/3), activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        with open('models/training_data_positive.clf', 'rb') as file:
            X = pickle.load(file)
        data1 = np.vstack(X)
        with open('models/training_data_negative.clf', 'rb') as file:
            X1 = pickle.load(file)
        data2 = np.vstack(X1)
        data = np.vstack((data1, data2))
        y = list()
        for i in range(len(X)):
            y.append(1)
        for i in range(len(X1)):
            y.append(0)
        y = np.asarray(y)
        model.fit(data, y, epochs=200, batch_size=5)
        model.save('models/predictor_NN_model.h5')

        return model

    def predict(self, image_path, threshold=0.85):
        utils = IU()
        model = load_model('models/predictor_NN_model.h5')
        image = cv2.imread(image_path)
        locs = utils.get_face_locations(image=image,
                                        model=self.face_model,
                                        scaleup=self.upsample)
        encodings = list()
        for (sX, sY, eX, eY) in locs:
            face = cv2.resize(image[sY:eY, sX:eX], (160, 160))
            encoding = utils.face_encodings(face,
                                            model=self.encoding_model,
                                            jitters=self.jitters)
            if len(encoding) == 0:
                continue
            else:
                encodings.append(encoding[0])
        if len(encodings) == 0:
            return False
        data = np.vstack(encodings)
        predictions = model.predict(data)
        for prediction in predictions:
            if prediction[0] > threshold:
                return True

        return False

    def get_image_list(self):
        images_list = list()
        try:
            for path in os.listdir(self.folder):
                if re.match('.*\.(jpg|png)', path.lower()):
                    images_list.append(os.path.join(self.folder, path))
        except:
            print("Error - folder or file path is invalid")
            sys.exit(0)
        images_list.sort()
        return images_list
