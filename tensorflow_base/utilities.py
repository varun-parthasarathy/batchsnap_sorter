import cv2
import numpy as np
from shutil import rmtree
import os
import pickle
import tensorflow as tf
from PIL import Image
from facenet import facenet
from facenet.align import detect_face


class ImageUtilities(object):

    def __init__(self):
        self.detector_net = cv2.dnn.readNetFromCaffe('models/deploy.prototxt.txt',
                                                     'models/ssd_detector.caffemodel')

    def get_face_locations(self, image, conf=0.7):
        locs = list()
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)),
                                     1.0,
                                     (300, 300),
                                     (104.0, 177.0, 123.0))
        self.detector_net.setInput(blob)
        detections = self.detector_net.forward()
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype('int')
                if startX < 0:
                    startX = 0
                if startY < 0:
                    startY = 0
                locs.append((startX, startY, endX, endY))

        return locs

    def generate_training_set(self):
        video_capture = cv2.VideoCapture(0)
        count = 0
        encodings = list()
        while True:
            ret, frame = video_capture.read()
            if not ret:
                continue
            orig = frame.copy()
            locations = self.get_face_locations(frame)
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
                encoding = self.face_encodings(face)
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

    def face_encodings(self, image, prewhiten=True, align=False):
        with tf.Graph().as_default():
            with tf.Session() as session:
                facenet.load_model('models/20180402-114759.pb')
                image = cv2.resize(image, (160, 160))
                image = image[:, :, ::-1]
                if align is True:
                    image = self.align_face(image)
                if prewhiten is True:
                    image = self.prewhiten(image)
                img_holder = tf.get_default_graph().get_tensor_by_name(
                    'input:0')
                embeddings = tf.get_default_graph().get_tensor_by_name(
                    'embeddings:0')
                phase_train = tf.get_default_graph().get_tensor_by_name(
                    'phase_train:0')
                feed_dict = {img_holder:[image], phase_train:False}
                encoding = session.run(embeddings, feed_dict = feed_dict)

        return encoding

    def prewhiten(self, x):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1/std_adj)
        
        return y

    def align_face(self, image, margin=44, size=182):
        minsize = 20
        threshold = [0.6, 0.7, 0.7]
        factor = 0.709
        with tf.Graph().as_default():
            session = tf.Session()
            with session.as_default():
                pnet, rnet, onet = detect_face.create_mtcnn(session, None)
        bounding_boxes, _ = detect_face.detect_face(image, minsize, pnet,
                                                    rnet, onet,
                                                    threshold,
                                                    factor)
        if bounding_boxes.shape[0] > 0:
            det = bounding_boxes[:, 0:4]
            det_arr = []
            img_size = np.asarray(image.shape)[0:2]
            det_arr.append(np.squeeze(det))

            for i, det in enumerate(det_arr):
                det = np.squeeze(det)
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0] - margin/2, 0)
                bb[1] = np.maximum(det[1] - margin/2, 0)
                bb[2] = np.minimum(det[2] + margin/2, img_size[1])
                bb[3] = np.minimum(det[3] + margin/2, img_size[0])
                cropped = image[bb[1]:bb[3], bb[0]:bb[2], :]
                scaled = cv2.resize(cropped, (size, size))

        aligned = np.asarray(scaled)

        return aligned
                
