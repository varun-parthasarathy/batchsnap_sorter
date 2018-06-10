import os
import cv2
import sys
import glob
import math
import pickle
import numpy as np
from PIL import Image
from shutil import rmtree
from sklearn import neighbors
import face_recognition as FR
from face_recognition.face_recognition_cli import image_files_in_folder


class KNNSorter(object):

    def __init__(self):
        self.folder = None
        self.name = 'search_face'
        self.ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

    def set_folder(self, folder):
        self.folder = folder

    def generator(self):
        video_capture = cv2.VideoCapture(0)
        count = 0
        personal_path = "/".join(['__training_data__', self.name])
        try:
            rmtree(personal_path)
        except:
            pass
        if not os.path.exists('__training_data__'):
            os.makedirs('__training_data__')
        if not os.path.exists(personal_path):
            os.makedirs(personal_path)
        while True:
            ret, frame = video_capture.read()
            if not ret:
                continue
            color = frame[:, :, ::-1]
            face_locations = FR.face_locations(color, model="hog")
            for (top, right, bottom, left) in face_locations:
                w = right - left; x = left; h = bottom - top; y = top; 
                img = cv2.rectangle(frame, (x, y), (x+w, y+h),
                                    (0, 255, 0), 2)
                f = cv2.resize(frame[y:y+h, x:x+w], (200, 200))
                file = "/".join([personal_path, str(count)+".png"])
                cv2.imwrite(file, f)
                count += 1
            cv2.imshow('Webcam Feed', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        video_capture.release()
        cv2.destroyAllWindows()

    def train(self, n_neighbors=None, knn_algo='ball_tree'):
        X = []
        y = []
        train_dir = '__training_data__'
        model_save_path = "predictor_model.clf"
        for class_dir in os.listdir(train_dir):
            if not os.path.isdir(os.path.join(train_dir, class_dir)):
                continue
            for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
                image = FR.load_image_file(img_path)
                face_bounding_boxes = FR.face_locations(image)

                if len(face_bounding_boxes) != 1:
                    pass
                else:
                    X.append(FR.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                    y.append(class_dir)
        if n_neighbors is None:
            n_neighbors = int(round(math.sqrt(len(X))))
        knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
        knn_clf.fit(X, y)
        if model_save_path is not None:
            with open(model_save_path, 'wb') as f:
                pickle.dump(knn_clf, f)
        return knn_clf

    def predict(self, X_img_path, knn_clf=None, distance_threshold=0.6, resize=True):
        model_path = "predictor_model.clf"
        if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in self.ALLOWED_EXTENSIONS:
            raise Exception("Invalid image path: {}".format(X_img_path))
        if knn_clf is None and model_path is None:
            raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")
        if knn_clf is None:
            with open(model_path, 'rb') as f:
                knn_clf = pickle.load(f)
        if resize is True:
            X_img = Image.open(X_img_path)
            size = X_img.size
            X_img = X_img.resize((int(size[0]*0.25), int(size[1]*0.25)),
                                 Image.ANTIALIAS)
            X_img = np.array(X_img)
        else:
            X_img = FR.load_image_file(X_img_path)
        X_face_locations = FR.face_locations(X_img)
        if len(X_face_locations) == 0:
            return []
        faces_encodings = FR.face_encodings(X_img, known_face_locations=X_face_locations)
        closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
        are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
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


class FaceIdentifier(object):

    def __init__(self, test_image=None, locations=None, tolerance=0.6):
        self.test_encoding = FR.face_encodings(test_image,
                                               locations)[0]
        self.result_images = list()
        self.tolerance = tolerance

    def encode_faces(self, path):
        try:
            image = Image.open(path)
        except:
            print("Error - Specified image does not exist")
            return
        size = image.size
        image = image.resize((int(size[0]*0.25), int(size[1]*0.25)),
                                       Image.ANTIALIAS)
        image = np.array(image)
        locs = FR.face_locations(image,
                                 number_of_times_to_upsample=1,
                                 model="hog")
        list_of_faces = FR.face_encodings(image, locs)
        res = self.compare_faces(list_of_faces, path)
        return

    def compare_faces(self, list_of_faces, path):
        if len(list_of_faces) < 1:
            return False
        result = FR.compare_faces(list_of_faces, self.test_encoding,
                                  tolerance=self.tolerance)
        if True in result:
            self.result_images.append(path)
            return True
        else:
            return False

    def get_results(self):
        results = self.result_images
        self.result_images = list()
        return results

    def get_image_list(self, folder_path):
        images_list = list()
        try:
            for path in glob.glob(os.path.join(folder_path, "*.jpg")):
                images_list.append(path)
        except:
            print("Error - folder or file path is invalid")
            sys.exit(0)
        return images_list
