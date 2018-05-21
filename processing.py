import face_recognition
import os
import glob
from skimage import io


class FaceIdentifier(object):

    def __init__(self, test_image_path = None):
        test_image = face_recognition.load_image_file(test_image_path)
        self.test_encode = face_recognition.face_encodings(test_image)[0]
        self.result_images = list()

    def encode_faces(self, path, result):
        image = face_recognition.load_image_file(path)
        list_of_faces = face_recognition.face_encodings(image)
        res = self.compare_faces(list_of_faces, path)
        if res is True:
            result.append(path)

    def compare_faces(self, list_of_faces, path):
        if len(list_of_faces) < 1:
            return False
        for face in list_of_faces:
            result = face_recognition.compare_faces([self.test_encode], face)
            if True in result:
                self.result_images.append(path)
                return True
            else:
                continue
        return False

    def load_images(self, folder_path):
        for path in glob.glob(os.path.join(folder_path, "*.jpg")):
            self.encode_faces(path)
        return self.result_images

    def get_image_list(self, folder_path):
        images_list = list()
        for path in glob.glob(os.path.join(folder_path, "*.jpg")):
            images_list.append(path)
        return images_list
            