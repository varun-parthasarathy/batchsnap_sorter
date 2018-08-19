from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import os
import sys
import pickle
from shutil import copy
from batch_sorter import BatchSorter


class ImageSorter(QWidget):

    def __init__(self, parent=None):
        super(ImageSorter, self).__init__(parent)
        self.setGeometry(500, 500, 300, 300)
        self.setWindowTitle('ImageSorter')
        self.box = QVBoxLayout()
        self.folder = None
        self.sort_path = None

        self.label1 = QLabel()
        self.label2 = QLabel()
        self.status = QLabel()
        self.progress = QProgressBar()
        self.label3 = QLabel()
        self.button1 = QPushButton('Select folder containing images to sort')
        self.button2 = QPushButton('Where should the sorted images be stored?')
        self.button1.clicked.connect(self.get_folder_path)
        self.button2.clicked.connect(self.get_sorted_path)
        self.button4 = QPushButton('Sort Images')
        self.button4.clicked.connect(self.sort_images)

        self.box.addWidget(self.button1)
        self.box.addWidget(self.label1)
        self.box.addWidget(self.button2)
        self.box.addWidget(self.label2)
        self.box.addWidget(self.button4)
        self.box.addWidget(self.progress)
        self.box.addWidget(self.status)

        self.setLayout(self.box)

    def get_folder_path(self):
        folder_path = QFileDialog()
        folder_path.setFileMode(QFileDialog.Directory)
        folder = None
        if folder_path.exec_():
            folder = folder_path.selectedFiles()
        if folder:
            self.folder = folder[0]
            self.label1.setText(self.folder)

    def get_sorted_path(self):
        sort_path = QFileDialog()
        sort_path.setFileMode(QFileDialog.Directory)
        sort = None
        if sort_path.exec_():
            sort = sort_path.selectedFiles()
        if sort:
            self.sort_path = sort[0]
            self.label2.setText(self.sort_path)

    def sort_images(self):
        if self.folder is None or self.sort_path is None:
            error = QErrorMessage()
            error.setWindowTitle('Error')
            error.showMessage('One or more paths have not been set')
            error.exec_()
        else:
            self.progress.setValue(0)
            self.status.setText('Encoding image data...')
            batcher = BatchSorter(self.folder)
            batcher.create_data_points(self.progress)
            self.progress.setValue(100)
            self.status.setText("Clustering data... Press 'k' to keep a given cluster, 'n' to move to the next one")
            self.progress.setValue(0)
            images = batcher.cluster(self.progress)
            self.progress.setValue(100)
            self.status.setText('Copying images...')
            self.progress.setValue(0)
            done = 0
            if len(images) > 0:
                increment = float(100.00/len(images))
                for image in images:
                    copy(image, self.sort_path)
                    done += increment
                    self.progress.setValue(done)
            self.progress.setValue(100)
            self.status.setText('Done!')


def main():
    app = QApplication(sys.argv)
    UI = ImageSorter()
    UI.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
