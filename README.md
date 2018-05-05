# batchsnap_sorter
App to sort batch snaps using face recognition

This app allows users to filter through hundreds of assorted photos and pick out their photos easily, using face recognition techniques.
The user must select a photo that clearly displays their face, and then select the folder in which the rest of the images are stored.
Once this is done, the app will sort the photos and copy them to a temporary folder in the same path as the app itself.

The app the face_recognition module, which in turn uses dlib, allowing the accuracy of the sorting to be over 95%.
The sorting is still time-consuming, and therefore plans are being made to use multithreading to allow for quicker sorting.
