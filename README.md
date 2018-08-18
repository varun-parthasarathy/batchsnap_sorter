# batchsnap_sorter
App to sort photos using face recognition

This app allows users to filter through hundreds of assorted photos and pick out their photos easily, by applying face recognition procedures.

## Requirements
The main requirement for this app is a webcam, as well as the latest version of the following modules, which can be installed with `pip3`-
```
dlib
opencv-python>=3.3
face_recognition
pyqt5
scikit-learn
numpy
tensorflow / tensorflow-gpu (if a GPU is present)
```
You will also need the latest version of FaceNet, which can be found [here](https://github.com/davidsandberg/facenet).
Additionally, to use the code in the `nn` folder, you must install `keras` as well. You can also compile `dlib` with GPU support; this
will make the code run much faster.

## Usage
Once you have all the dependancies installed, you can run the app using-
`python3 main.py`

This will open a GUI, with which you can sort your photos and set sorting options. The first time you use the app, you must train the classifiers to recognize your face. To do so, click on 'Advanced Options', then look for the button that says 'Generate training data' and click on it. A webcam feed will open, and a box will appear around your face with a counter. Press 'k' to capture an image of your face. Capture atleast 30 photos, then press 'q' to close the feed. Next, press 'Train classifier'. There are 3 classifiers available; each one must be trained separately as of now. The classifier that will be used is up to you.
Once the classifier is trained, you're set! The classifiers don't have to be trained again, unless you want to detect someone else's face. You can modify other options as well, such as the distance metric, the face detection model, the face encoding model, and the filters that can be applied to your photos. You can even filter your photos without sorting them.
To run the sorter, select the folder containing the images to sort, and the folder in which you want to save the result photos. Once that's done, click on 'Sort Images' to start. Depending on the advanced options you've set, this could take a while. Once it's done, it will copy the images to the results folder you specified, and then you'll see the status bar display 'Done!'

## The `nn` folder
The files in the `nn` folder provide an alternate method to sort the images accurately, although it's slightly trickier to set up. For this, the module `keras` is required, which can be installed using `pip`.
This method uses a neural network to determine if a face matches the users' or not. It runs very slowly without a GPU, but can give extremely accurate results.

To set it up, you must generate two types of training data - positive data, which is basically images of your face, and negative data, which is images of people who are not you. For best results, generate negative data from the faces of family members and people who look similar to you or who wear glasses, as the usual classification options struggle with the similarities between family, as well as people with glasses. Once this is done, train the neural network classifier. Then you can use the application as described above. 

The script `main_dlib.py` does the same thing as `main.py`, except that it only uses dlib's implementation of a face embeddings generator, instead of the FaceNet generator, and as a result it is slightly faster.

### Note on `alternate_generator.py`
This script allows you to generate training data without a webcam. All you have to do is select images of yourself from your computer, and click on 'Generate training data'. Once this is done, you can run `main.py` and train your classifiers.

### Advanced settings
Once you are familiar with the interface, you can make more changes to the application settings, such as-  
_Face detection model_ - The face detection model to use. Default is HOG+SVM, which is faster than the CNN method, but is less accurate.                          The CNN method is best suited to machines that have a GPU and enough RAM (atleast 6 GB)  
_Face encoding model_  - The face encoding model to use. Default is dlib's 128D model. The FaceNet model generates 512D encodings of 
                         faces instead, which gives greater accuracy, but as expected, is much slower. The 128D model and the 512D model 
                         are incompatible with each other, so if you change this setting, you will need to recreate training data and
                         retrain the classifiers.  
_Random operations_    - Specify the number of random operations to be performed on each face. Default is 3. Greater the number, higher
                         the accuracy, and slower the code. For example, a value of 10 makes the code ten times slower.  
_Scale up image_       - Number of times to scale up the image while looking for faces. Default is 1. This allows smaller faces to be
                         detected in an image.  
_Error threshold_      - Set the error threshold for each classifier. If using the 512D model, the thresholds must be made double their
                         previous value. A larger threshold means more false positives, while a lower threshold restricts the number of
                         matches found.
                         
Once you are done changing the settings, click on 'Set Options' to set them, then use the app as usual.

## Author
[Varun Parthasarathy](https://github.com/Var-ji)
