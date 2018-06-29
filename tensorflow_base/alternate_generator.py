from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import os
import sys
import pickle
import numpy as np
import face_recognition as FR
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
        self.generate = QPushButton('Generate training data')
        self.button.clicked.connect(self.select_images)
        self.generate.clicked.connect(self.generate_training_set)
        self.info.setText('''This app can be used to generate a training set
        using pre-existing images. This is helpful when the
        quality of your webcam isn't particularly good. Just
        make sure that the photos you use have your face only;
        if there are multiple faces, the first one detected
        will be used. Atleast 10 photos are required to get
        a good output.''')
        self.box.addWidget(self.info)
        self.box.addWidget(self.button)
        self.box.addWidget(self.images)
        self.box.addWidget(self.generate)
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

    def generate_training_set(self):
        if self.paths is None or len(self.paths) < 10:
            error = QErrorMessage()
            error.showMessage('Select atleast 10 images')
            error.exec_()
        else:
            utils = IU()
            self.status.setText('Generating training data...')
            encodings = list()
            for image_path in self.paths:
                image = cv2.imread(image_path)
                (sX, sY, eX, eY) = utils.get_face_locations(image=image)[0]
                face = cv2.resize(image[sY:eY, sX:eX], (200, 200))
                face = face[:, :, ::-1]
                encoding = utils.face_encodings(face, align=True)
                if len(encoding) == 0:
                    pass
                else:
                    encodings.append(encoding[0])
            with open('models/training_data.clf', 'wb') as file:
                pickle.dump(encodings, file)
            self.status.setText('Done!')

def main():
    app = QApplication(sys.argv)
    UI = AlternateGenerator()
    UI.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
            
