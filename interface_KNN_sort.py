from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import os
import sys
from processing import KNNSorter
from shutil import copy
import threading
import face_recognition as FR
from PIL import Image
import numpy as np


class PictureSorter(QWidget):

    def __init__(self, parent=None):
        super(PictureSorter, self).__init__(parent)
        self.identifier = KNNSorter()
        self.setGeometry(600, 600, 300, 300)
        self.setWindowTitle('PicSorter')
        self.box = QVBoxLayout()
        self.folder = None
        self.sort_path = None
        self.l2 = QLabel()
        self.l4 = QLabel()
        self.status = QLabel()
        self.progress = QProgressBar()
        self.tolerance = 0.6
        valid = QDoubleValidator()
        valid.setRange(0.005, 1.000, 3)
        self.textbox = QLineEdit()
        self.textbox.setText('0.6')
        self.textbox.setValidator(valid)
        self.l3 = QLabel()
        self.l3.setText('''Set tolerance - a higher tolerance will
                           identify more photos, but may have more
                           false positives. A lower tolerance will
                           be more accurate, but all the photos may
                           not be identified accurately. Values range
                           from 0.005 to 1''')
        self.b2 = QPushButton("Select folder containing images to sort")
        self.bgen = QPushButton("Generate training set of images")
        self.bgen.clicked.connect(self.generate_training_set)
        self.btrain = QPushButton("Train predictor model")
        self.btrain.clicked.connect(self.train_classifier)
        self.button = QPushButton("Sort Images")
        self.button.clicked.connect(self.sort_images)
        self.b2.clicked.connect(self.get_folder_path)
        self.b3 = QPushButton("Where should the sorted images be stored?")
        self.b3.clicked.connect(self.get_sorted_path)
        self.box.addWidget(self.b2)
        self.box.addWidget(self.l2)
        self.box.addWidget(self.l3)
        self.box.addWidget(self.textbox)
        self.box.addWidget(self.b3)
        self.box.addWidget(self.l4)
        self.box.addWidget(self.bgen)
        self.box.addWidget(self.btrain)
        self.box.addWidget(self.button)
        self.box.addWidget(self.progress)
        self.box.addWidget(self.status)
        self.setLayout(self.box)

    def generate_training_set(self):
        self.identifier.generator()

    def train_classifier(self):
        if os.path.isdir('__training_data__/search_face') and len(os.listdir('__training_data__/search_face')) > 0:
            self.identifier.train()
        else:
            pass

    def get_folder_path(self):
        folder_path = QFileDialog()
        folder_path.setFileMode(QFileDialog.Directory)
        folder = None
        if folder_path.exec_():
            folder = folder_path.selectedFiles()
        if folder:
            self.folder = folder[0]
            self.l2.setText(self.folder)

    def get_sorted_path(self):
        sort_path = QFileDialog()
        sort_path.setFileMode(QFileDialog.Directory)
        sort = None
        if sort_path.exec_():
            sort = sort_path.selectedFiles()
        if sort:
            self.sort_path = sort[0]
            self.l4.setText(self.sort_path)

    def sort_images(self):
        if self.folder is None or self.sort_path is None:
            pass
        elif self.textbox.text() == "":
            pass
        elif os.path.isfile('predictor_model.clf'):
            self.progress.setValue(0)
            done = 0
            self.status.setText('Sorting images...')
            self.tolerance = float(self.textbox.text())
            self.identifier.set_folder(self.folder)
            image_list = self.identifier.get_image_list()
            increment = float(100.00/float(len(image_list)))
            results = list()
            for image in image_list:
                res = self.identifier.predict(X_img_path=image,
                                              distance_threshold=self.tolerance)
                for name, (x, y, w, h) in res:
                    if name.lower() == 'search_face':
                        results.append(image)
                        break
                    else:
                        res = self.identifier.predict(X_img_path=image,
                                                      distance_threshold=self.tolerance,
                                                      resize=False)
                        for name, (x, y, w, h) in res:
                            if name.lower() == 'search_face':
                                results.append(image)
                                break
                            else:
                                pass
                done += increment
                self.progress.setValue(done)
            self.progress.setValue(100)
            if not os.path.exists(self.sort_path):
                os.makedirs(self.sort_path)
            self.progress.setValue(0); done = 0;
            self.status.setText('Copying results to folder...')
            for image in results:
                copy(image, self.sort_path)
                done += increment
                self.progress.setValue(done)
            self.progress.setValue(100)
            self.status.setText('Done!')
        else:
            pass


def main():
    app = QApplication(sys.argv)
    UI = PictureSorter()
    UI.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
