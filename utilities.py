import cv2
import numpy as np
import dlib
from shutil import rmtree
import os
import pickle
import tensorflow as tf
from PIL import Image
from facenet import facenet
import face_recognition_models as FRM
import face_recognition as FR


class ImageUtilities(object):

    def __init__(self):
        pass

    def get_face_locations(self, image, model='hog', scaleup=1):
        locs = list()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        (h, w) = image.shape[:2]
        image = cv2.resize(image, (int(w*0.4), int(h*0.4)))
        faces = FR.face_locations(image,
                                  number_of_times_to_upsample=scaleup,
                                  model=model)
        for (t, r, b, l) in faces:
            sX = l; sY = t; eX = r; eY = b;
            locs.append((sX, sY, eX, eY))

        return locs

    def generate_training_set(self, encoder='128D', jitters=3,
                              face_model='hog', scaleup=1):
        video_capture = cv2.VideoCapture(0)
        count = 0
        encodings = list()
        while True:
            ret, frame = video_capture.read()
            if not ret:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            orig = frame.copy()
            locations = self.get_face_locations(frame,
                                                model=face_model,
                                                scaleup=scaleup)
            text = 'Count = ' + str(count)
            for (sX, sY, eX, eY) in locations:
                cv2.rectangle(frame, (sX, sY),
                              (eX, eY),
                              (0, 255, 0), 2)
                y = sY - 10 if sY - 10 > 10 else sY + 10
                cv2.putText(frame, text,
                            (sX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            cv2.imshow('Webcam Feed', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('k'):
                face = cv2.resize(orig[sY:eY, sX:eX], (200, 200))
                encoding = self.face_encodings(face,
                                               model=encoder,
                                               jitters=jitters)
                if len(encoding) == 0:
                    pass
                else:
                    encodings.append(encoding[0])
                    count += 1
            elif key == ord('q') and count > 30:
                with open('models/training_data.clf', 'wb') as file:
                    pickle.dump(encodings, file)
                break
            # Debug
            elif key == ord('d'):
                with open('models/training_data.clf', 'wb') as file:
                    pickle.dump(encodings, file)
                break
        video_capture.release()
        cv2.destroyAllWindows()

    def detect_objects(self, image_list, conf=0.4, bar=None, classes=None):
        net = cv2.dnn.readNetFromCaffe('models/MNSSD_deploy.prototxt.txt',
                                       'models/MNSSD_detector.caffemodel')
        CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
        results = list()
        done = 0
        increment = float(100.00/len(image_list))
        if bar is not None:
            bar.setValue(0)
        for image_path in image_list:
            image = cv2.imread(image_path)
            (h, w) = image.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)),
                                         0.007843,
                                         (300, 300),
                                         127.5)
            net.setInput(blob)
            detections = net.forward()
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > conf:
                    idx = int(detections[0, 0, i, 1])
                    if CLASSES[idx] in classes:
                        results.append(image_path)
                        break
            if bar is not None:
                done += increment
                bar.setValue(done)
        if bar is not None:
            bar.setValue(100)

        return results

    def face_encodings(self, image, model='128D', jitters=3,
                       prewhiten=True, align=False):
        image = cv2.resize(image, (160, 160))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if align is True:
            image = self.align_face(image)
        if model == '512D':
            if prewhiten is True:
                image = self.prewhiten(image)
            with tf.Graph().as_default():
                with tf.Session() as session:
                    facenet.load_model('models/20180402-114759.pb')
                    img_holder = tf.get_default_graph().get_tensor_by_name(
                        'input:0')
                    embeddings = tf.get_default_graph().get_tensor_by_name(
                        'embeddings:0')
                    phase_train = tf.get_default_graph().get_tensor_by_name(
                        'phase_train:0')
                    feed_dict = {img_holder:[image], phase_train:False}
                    encoding = session.run(embeddings, feed_dict = feed_dict)
        else:
            encoding = FR.face_encodings(image,
                                         num_jitters=jitters)

        return encoding

    def prewhiten(self, x):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1/std_adj)
        
        return y

    def align_face(self, image):
        faces = dlib.full_object_detections()
        (h, w) = image.shape[:2]
        loc = dlib.rectangle(h, w, 0, 0)
        model = FRM.pose_predictor_five_point_model_location()
        sp = dlib.shape_predictor(model)
        faces.append(sp(image, loc))
        aligned = dlib.get_face_chip(image, faces[0],
                                     size=160)

        return aligned

    def face_distance(self, face_encodings, face_to_compare):
        if len(face_encodings) == 0:
            return np.empty((0))
        return np.linalg.norm(face_encodings - face_to_compare, axis = 1)

    def compare_faces(self, known_encodings, face_encoding, tolerance=1.2):
        distances = self.face_distance(known_encodings,
                                       face_encoding)
        return list(distances <= tolerance)

