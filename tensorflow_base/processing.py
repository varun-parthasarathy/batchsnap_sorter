import os
import cv2
import sys
import glob
import math
import pickle
import numpy as np
from sklearn import neighbors, svm
import face_recognition as FR
from utilities import ImageUtilities as IU


class KNNSorter(object):

    def __init__(self):
        self.folder = None
        self.utils = IU()

    def set_folder(self, folder):
        self.folder = folder

    def train(self, n_neighbors=None, knn_algo='ball_tree'):
        X = []
        y = []
        model_save_path = "models/predictor_knn_model.clf"
        with open('models/training_data.clf', 'rb') as file:
            X = pickle.load(file)
        for i in range(len(X)):
            y.append('search_face')
        if n_neighbors is None:
            n_neighbors = int(round(math.sqrt(len(X))))
        knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors,
                                                 algorithm=knn_algo,
                                                 weights='distance')
        knn_clf.fit(X, y)
        if model_save_path is not None:
            with open(model_save_path, 'wb') as f:
                pickle.dump(knn_clf, f)
        return knn_clf

    def predict(self, image_path, knn_clf=None, threshold=0.6):
        model_path = "models/predictor_knn_model.clf"
        if not os.path.isfile(image_path):
            raise Exception("Invalid image path: {}".format(X_img_path))
        if knn_clf is None and model_path is None:
            raise Exception("Must supply knn classifier either through knn_clf or model_path")
        if knn_clf is None:
            with open(model_path, 'rb') as f:
                knn_clf = pickle.load(f)
        X_img = cv2.imread(image_path)
        X_face_locations = self.utils.get_face_locations(image=X_img)
        if len(X_face_locations) == 0:
            return []
        faces_encodings = list()
        for (x, y, a, b) in X_face_locations:
            face = X_img[y:b, x:a]
            encode = self.utils.face_encodings(X_img, align=True)
            faces_encodings.append(encode[0])
        closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
        are_matches = [closest_distances[0][i][0] <= threshold for i in range(len(X_face_locations))]
        return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

    def get_image_list(self):
        images_list = list()
        try:
            for path in glob.glob(os.path.join(self.folder, "*.jpg")):
                images_list.append(path)
            for path in glob.glob(os.path.join(self.folder, "*.png")):
                images_list.append(path)
        except:
            print("Error - folder or file path is invalid")
            sys.exit(0)
        images_list.sort()
        return images_list


class SVMSorter(object):

    def __init__(self):
        self.folder = None
        self.utils = IU()

    def set_folder(self, folder):
        self.folder = folder

    def train(self):
        X = []
        y = []
        model_save_path = "models/predictor_svm_model.clf"
        with open('models/training_data.clf', 'rb') as file:
            X = pickle.load(file)
        for i in range(len(X)):
            y.append('search_face')
        svm_clf = svm.OneClassSVM()
        svm_clf.fit(X, y)
        if model_save_path is not None:
            with open(model_save_path, 'wb') as file:
                pickle.dump(svm_clf, file)
        return svm_clf

    def predict(self, image_path, svm_clf=None, threshold=0.006):
        model_path = "models/predictor_svm_model.clf"
        if threshold > 0:
            threshold *= -1
        if not os.path.isfile(image_path):
            raise Exception("Invalid image path: {}".format(X_img_path))
        if svm_clf is None and model_path is None:
            raise Exception("Must supply svm classifier either through svm_clf or model_path")
        if svm_clf is None:
            with open(model_path, 'rb') as f:
                svm_clf = pickle.load(f)
        X_img = cv2.imread(image_path)
        X_face_locations = self.utils.get_face_locations(image=X_img)
        if len(X_face_locations) == 0:
            return False
        faces_encodings = list()
        for (x, y, a, b) in X_face_locations:
            face = X_img[y:b, x:a]
            encode = self.utils.face_encodings(X_img, align=True)
            faces_encodings.append(encode[0])
        distances = svm_clf.decision_function(faces_encodings)
        for distance in distances:
            if distance < threshold or distance > 0:
                return True
        return False

    def get_image_list(self):
        images_list = list()
        try:
            for path in glob.glob(os.path.join(self.folder, "*.jpg")):
                images_list.append(path)
            for path in glob.glob(os.path.join(self.folder, "*.png")):
                images_list.append(path)
        except:
            print("Error - folder or file path is invalid")
            sys.exit(0)
        images_list.sort()
        return images_list


class EuclideanSorter(object):

    def __init__(self):
        self.utils = IU()
        self.folder = None

    def set_folder(self, folder):
        self.folder = folder

    def train(self):
        model_save_path = "models/predictor_euclidean_model.clf"
        with open('models/training_data.clf', 'rb') as file:
            encodings = pickle.load(file)
        encodings = np.array(encodings)
        mean_encoding = encodings.mean(axis=0)
        if model_save_path is not None:
            with open(model_save_path, 'wb') as file:
                pickle.dump(mean_encoding, file)
        return mean_encoding

    def predict(self, image_path, threshold=0.6):
        model_path = "models/predictor_euclidean_model.clf"
        model_encoding = None
        if not os.path.isfile(image_path):
            raise Exception("Invalid image path: {}".format(image_path))
        if model_encoding is None and model_path is None:
            raise Exception("Must supply svm classifier either through svm_clf or model_path")
        if model_encoding is None:
            with open(model_path, 'rb') as f:
                model_encoding = pickle.load(f)
        image = cv2.imread(image_path)
        locs = self.utils.get_face_locations(image=image)
        if len(locs) == 0:
            return False
        faces_encodings = list()
        for (x, y, a, b) in locs:
            face = image[y:b, x:a]
            encode = self.utils.face_encodings(image, align=True)
            faces_encodings.append(encode[0])
        results = FR.compare_faces(encodings,
                                   model_encoding,
                                   tolerance=threshold)
        if True in results:
            return True
        else:
            return False

    def get_image_list(self):
        images_list = list()
        try:
            for path in glob.glob(os.path.join(self.folder, "*.jpg")):
                images_list.append(path)
            for path in glob.glob(os.path.join(self.folder, "*.png")):
                images_list.append(path)
        except:
            print("Error - folder or file path is invalid")
            sys.exit(0)
        images_list.sort()
        return images_list

    
