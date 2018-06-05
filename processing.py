import face_recognition as FR
import os
import glob
from PIL import Image
import numpy as np


class FaceIdentifier(object):

    def __init__(self, test_image_path = None):
        test_image = Image.open(test_image_path)
        size = test_image.size
        test_image = test_image.resize((int(size[0]*0.25), int(size[1]*0.25)),
                                       Image.ANTIALIAS)
        #test_image = test_image.convert('L')
        test_image = np.array(test_image)
        self.test_locations = FR.face_locations(test_image,
                                                number_of_times_to_upsample=1,
                                                model="hog")
        self.test_encoding = FR.face_encodings(test_image,
                                               self.test_locations)[0]
        self.result_images = list()

    def encode_faces(self, path):
        image = Image.open(path)
        size = image.size
        image = image.resize((int(size[0]*0.25), int(size[1]*0.25)),
                                       Image.ANTIALIAS)
        #image = image.convert('L')
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
                                  tolerance=0.5)
        if True in result:
            self.result_images.append(path)
            return True
        else:
            return False

    def get_results(self):
        return self.result_images

    def get_image_list(self, folder_path):
        images_list = list()
        for path in glob.glob(os.path.join(folder_path, "*.jpg")):
            images_list.append(path)
        return images_list
            
