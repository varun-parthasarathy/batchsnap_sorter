from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import os
import sys
from processing import *
from shutil import copy
import threading


class PictureSorter(QWidget):

    def __init__(self, parent = None):
        super(PictureSorter, self).__init__(parent)
        self.setGeometry(600, 600, 300, 300)
        self.setWindowTitle('PicSorter')
        self.box = QVBoxLayout()
        self.image_path = None
        self.folder = None
        self.b1 = QPushButton("Select test image")
        self.l1 = QLabel()
        self.l2 = QLabel()
        self.b2 = QPushButton("Select folder containing images to sort")
        self.button = QPushButton("Sort Images")
        self.button.clicked.connect(self.sort_images)
        self.b1.clicked.connect(self.get_test_path)
        self.b2.clicked.connect(self.get_folder_path)
        self.box.addWidget(self.b1)
        self.box.addWidget(self.l1)
        self.box.addWidget(self.b2)
        self.box.addWidget(self.l2)
        self.box.addWidget(self.button)
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

    def sort_images(self):
        if self.folder is None or self.image_path is None:
            pass
        else:
            identifier = FaceIdentifier(self.image_path)
            image_list = identifier.get_image_list(self.folder)
            image_list.sort()
            results = list()
            for path in image_list:
                print(path)
                identifier.encode_faces(path)
            results = identifier.get_results()
            if not os.path.exists("__sorted_images__"):
                os.makedirs("__sorted_images__")
            results.sort()
            for image in results:
                copy(image, "__sorted_images__")


def main():
    app = QApplication(sys.argv)
    UI = PictureSorter()
    UI.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
