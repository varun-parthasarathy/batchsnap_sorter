from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import os
import sys
from processing import *
from shutil import copy
import threading
import face_recognition as FR
from PIL import Image
import numpy as np


class PictureSorter(QWidget):

    def __init__(self, parent=None):
        super(PictureSorter, self).__init__(parent)
        self.setGeometry(600, 600, 300, 300)
        self.setWindowTitle('PicSorter')
        self.box = QVBoxLayout()
        self.image_path = None
        self.folder = None
        self.sort_path = None
        self.b1 = QPushButton("Select test image")
        self.l1 = QLabel()
        self.l2 = QLabel()
        self.l4 = QLabel()
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
        self.button = QPushButton("Sort Images")
        self.button.clicked.connect(self.sort_images)
        self.b1.clicked.connect(self.get_test_path)
        self.b2.clicked.connect(self.get_folder_path)
        self.b3 = QPushButton("Where should the sorted images be stored?")
        self.b3.clicked.connect(self.get_sorted_path)
        self.box.addWidget(self.b1)
        self.box.addWidget(self.l1)
        self.box.addWidget(self.b2)
        self.box.addWidget(self.l2)
        self.box.addWidget(self.l3)
        self.box.addWidget(self.textbox)
        self.box.addWidget(self.b3)
        self.box.addWidget(self.l4)
        self.box.addWidget(self.button)
        self.box.addWidget(self.progress)
        self.setLayout(self.box)

    def get_test_path(self):
        test_path = QFileDialog()
        test_path.setFileMode(QFileDialog.ExistingFile)
        image_path = None
        if test_path.exec_():
            image_path = test_path.selectedFiles()
        if image_path:
            self.image_path = image_path[0]
            self.l1.setText(self.image_path)

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
        if self.folder is None or self.image_path is None or self.sort_path is None:
            pass
        elif self.textbox.text() == "":
            pass
        else:
            try:
                test_image = Image.open(self.image_path)
            except:
                print("Error - Specified image does not exist")
                sys.exit(0)
            size = test_image.size
            test_image = test_image.resize((int(size[0]*0.25), int(size[1]*0.25)),
                                           Image.ANTIALIAS)
            test_image = np.array(test_image)
            locations = FR.face_locations(test_image,
                                               number_of_times_to_upsample=1,
                                               model="hog")
            if len(locations) < 1:
                print("There were no detectable faces in the image")
                sys.exit(0)
            complete = 0
            self.progress.setValue(complete)
            self.tolerance = float(self.textbox.text())
            identifier = FaceIdentifier(test_image,
                                        locations,
                                        self.tolerance)
            image_list = identifier.get_image_list(self.folder)
            image_list.sort()
            increment = float(100.00/float(len(image_list)))
            results = list()
            for path in image_list:
                identifier.encode_faces(path)
                complete += increment
                self.progress.setValue(complete)
            complete += increment
            self.progress.setValue(complete)
            results = identifier.get_results()
            if not os.path.exists(self.sort_path):
                os.makedirs(self.sort_path)
            results.sort()
            for image in results:
                copy(image, self.sort_path)


def main():
    app = QApplication(sys.argv)
    UI = PictureSorter()
    UI.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
