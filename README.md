# batchsnap_sorter
App to sort photos using face recognition

This app allows users to filter through hundreds of assorted photos and pick out their photos easily, by applying face recognition procedures.

There are two available methods for processing images. The KNN method takes photos of your face using your webcam (if available), and uses it as training data to create a classifier for your face. This can be done by pressing the "Generate training set of images" button. A webcam feed will show up, and a green box will appear around your face. Wait for a few seconds, then press 'q' to stop the feed. Once this is done, click on "Train predictor model" to set things up. Finally, click on "Sort Images" to sort your photos. This works slightly slower, as it reprocesses images when faces aren't found. The next time you use it, you can directly sort the images without having to use the generator or training the predictive model again.

The alternative is to simply provide a photo where your face is visible, preferably alone, and the sorter will sort the images by comparing the faces found in each one to the face you provided. This works comparatively faster than the KNN method, but results will vary depending on the photo you provide.

The KNN model generally gives better results if the photos are taken in good lighting conditions, but the alternative method is faster. The sort method is left up to the user.
