import face_recognition
import os
import glob
from skimage import io


class FaceIdentifier(object):

    def __init__(self, test_image_path = None):
        test_image = face_recognition.load_image_file(test_image_path)
        self.test_encode = face_recognition.face_encodings(test_image)[0]
        self.result_images = list()

    def encode_faces(self, path):
        image = face_recognition.load_image_file(path)
        list_of_faces = face_recognition.face_encodings(image)
        self.compare_faces(list_of_faces, path)

    def compare_faces(self, list_of_faces, path):
        if len(list_of_faces) < 1:
            return None
        for face in list_of_faces:
            result = face_recognition.compare_faces([self.test_encode], face)
            if True in result:
                self.result_images.append(path)
                return 0
            else:
                continue

    def load_images(self, folder_path):
        for path in glob.glob(os.path.join(folder_path, "*.jpg")):
            self.encode_faces(path)
        return self.result_images
