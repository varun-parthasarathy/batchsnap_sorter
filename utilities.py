import cv2
import face_recognition as FR
import numpy as np
from shutil import rmtree
import os
import pickle


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
                print('Bad frame!')
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
                face = face[:, :, ::-1]
                encoding = FR.face_encodings(face,
                                             num_jitters=10,
                                             known_face_locations=[(sX, sY, eX, eY)])
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
                
