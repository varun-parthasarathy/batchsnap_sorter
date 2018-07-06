from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import os
import sys
import pickle
import numpy as np
import face_recognition as FR
import cv2
from utilities import ImageUtilities as IU


class AlternateGenerator(QWidget):

    def __init__(self, parent=None):
        super(AlternateGenerator, self).__init__(parent)
        self.paths = None
        self.setGeometry(700, 700, 300, 300)
        self.setWindowTitle('Alternate Generator')
        self.box = QVBoxLayout()
        self.info = QLabel()
        self.images = QLabel()
        self.status = QLabel()
        self.button = QPushButton('Select images to use')
        self.generate_p = QPushButton('Generate positive training data')
        self.generate_n = QPushButton('Generate negative training data')
        self.button.clicked.connect(self.select_images)
        self.generate_p.clicked.connect(lambda *f: self.generate_training_set(model='+'))
        self.generate_n.clicked.connect(lambda *f: self.generate_training_set(model='-'))
        self.info.setText('''This app can be used to generate a training set
        using pre-existing images. This is helpful when the
        quality of your webcam isn't particularly good.
        Atleast 10 photos are required to get a good output.''')
        self.box.addWidget(self.info)
        self.box.addWidget(self.button)
        self.box.addWidget(self.images)
        self.box.addWidget(self.generate_p)
        self.box.addWidget(self.generate_n)
        self.box.addWidget(self.status)

        self.setLayout(self.box)

    def select_images(self):
        image_paths = QFileDialog()
        image_paths.setFileMode(QFileDialog.ExistingFiles)
        paths = None
        if image_paths.exec_():
            paths = image_paths.selectedFiles()
        if paths:
            self.paths = paths
            display = '\n'.join(paths)
            self.images.setText(display)

    def generate_training_set(self, model='+'):
        if model == '+':
            model_path = 'models/training_data_positive.clf'
        else:
            model_path = 'models/training_data_negative.clf'
        if self.paths is None or len(self.paths) < 10:
            error = QErrorMessage()
            error.setWindowTitle('Error')
            error.showMessage('Select atleast 10 images')
            error.exec_()
        else:
            utils = IU()
            self.status.setText('Generating training data...')
            encodings = list()
            for image_path in self.paths:
                image = cv2.imread(image_path)
                locs = utils.get_face_locations(image=image)
                for (sX, sY, eX, eY) in locs:
                    face = cv2.resize(image[sY:eY, sX:eX], (200, 200))
                    encoding = utils.face_encodings(face, model='128D',
                                                    jitters=10)
                    if len(encoding) == 0:
                        print('Bad image')
                        continue
                    else:
                        encodings.append(encoding[0])
            with open(model_path, 'wb') as file:
                pickle.dump(encodings, file)
            self.status.setText('Done!')

def main():
    app = QApplication(sys.argv)
    UI = AlternateGenerator()
    UI.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
            
