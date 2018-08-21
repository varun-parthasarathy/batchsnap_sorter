import cv2
import numpy as np
import face_recognition as FR
import pickle
import os
import re
from hdbscan import HDBSCAN
from sklearn.decomposition import PCA
from imutils import build_montages


class BatchSorter(object):

    def __init__(self, folder):
        self.folder = folder
        self.images_list = list()
        for path in os.listdir(self.folder):
            if re.match('.*\.(jpg|png)', path.lower()):
                self.images_list.append(os.path.join(self.folder, path))
        self.images_list.sort()

    def equalize(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l1 = clahe.apply(l)
        processed = cv2.merge((l1, a, b))
        processed = cv2.cvtColor(processed, cv2.COLOR_LAB2BGR)

        return processed

    def create_data_points(self, equalize=True, progress=None):
        data = list()
        done = 0
        increment = 0
        if progress is not None:
            increment = float(100.00/len(self.images_list))
            progress.setValue(0)
        for (i, path) in enumerate(self.images_list, 1):
            image = cv2.imread(path)
            print('[INFO] Processing image %d of %d' % (i, len(self.images_list)))
            (h, w) = image.shape[:2]
            image = cv2.resize(image, (int(w*0.25), int(h*0.25)))
            if equalize is True:
                image = self.equalize(image)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            boxes = FR.face_locations(rgb, model='hog',
                                      number_of_times_to_upsample=1)
            encodings = FR.face_encodings(rgb, boxes, num_jitters=10)
            done += increment
            if progress is not None:
                progress.setValue(done)

            if len(encodings) > 0:
                d = [{'path':path, 'loc':box, 'encoding':enc}
                     for (box, enc) in zip(boxes, encodings)]
                data.extend(d)

        with open('encoding_data.pkl', 'wb') as file:
            pickle.dump(data, file)

    def cluster(self, progress=None):
        images = list()
        with open('encoding_data.pkl', 'rb') as file:
            data = pickle.load(file)
        data = np.array(data)
        encodings = [d['encoding'] for d in data]
        X = np.vstack(encodings)
        pca = PCA(n_components='mle', svd_solver='full')
        X_new = pca.fit_transform(X)
        clt = HDBSCAN(metric='euclidean', min_cluster_size=5)
        clt.fit(X_new)

        labelIDs = np.unique(clt.labels_)
        done = 0
        increment = float(100.00/len(labelIDs))
        if progress is not None:
            progress.setValue(0)
        for labelID in labelIDs:
            faces = list()
            idxs = np.where(clt.labels_ == labelID)[0]
            idxs = np.random.choice(idxs, size=min(25, len(idxs)),
                                    replace=False)
            for i in idxs:
                image = cv2.imread(data[i]['path'])
                (h, w) = image.shape[:2]
                image = cv2.resize(image, (int(w*0.25), int(h*0.25)))
                (t, r, b, l) = data[i]['loc']
                face = image[t:b, l:r]
                face = cv2.resize(face, (96, 96))
                faces.append(face)

            montage = build_montages(faces, (96, 96), (5, 5))[0]
            if progress is not None:
                done += increment
                progress.setValue(done)
            title = 'Face ID #{}'.format(labelID)
            title = 'Unknown Faces' if labelID == -1 else title
            cv2.imshow(title, montage)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('k'):
                idxs = np.where(clt.labels_ == labelID)[0]
                for i in idxs:
                    images.append(data[i]['path'])
                cv2.destroyAllWindows()
            elif key == ord('n'):
                cv2.destroyAllWindows()

        return images
