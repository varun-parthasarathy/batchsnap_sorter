import face_recognition as FR
import os
import glob
import sys
from PIL import Image
import numpy as np


class FaceIdentifier(object):

    def __init__(self, test_image=None, locations=None, tolerance=0.6):
        self.test_encoding = FR.face_encodings(test_image,
                                               locations)[0]
        self.result_images = list()
        self.tolerance = tolerance

    def encode_faces(self, path):
        try:
            image = Image.open(path)
        except:
            print("Error - Specified image does not exist")
            return
        size = image.size
        image = image.resize((int(size[0]*0.25), int(size[1]*0.25)),
                                       Image.ANTIALIAS)
        image = np.array(image)
        locs = FR.face_locations(image,
                                 number_of_times_to_upsample=1,
                                 model="hog")
        list_of_faces = FR.face_encodings(image, locs)
        res = self.compare_faces(list_of_faces, path)
        return

    def compare_faces(self, list_of_faces, path):
        if len(list_of_faces) < 1:
            return False
        result = FR.compare_faces(list_of_faces, self.test_encoding,
                                  tolerance=self.tolerance)
        if True in result:
            self.result_images.append(path)
            return True
        else:
            return False

    def get_results(self):
        results = self.result_images
        self.result_images = list()
        return results

    def get_image_list(self, folder_path):
        images_list = list()
        try:
            for path in glob.glob(os.path.join(folder_path, "*.jpg")):
                images_list.append(path)
        except:
            print("Error - folder or file path is invalid")
            sys.exit(0)
        return images_list
