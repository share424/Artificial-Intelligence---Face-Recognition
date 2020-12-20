# Artificial Intelligence - Face Recognition
A simple face recognition project that use kNN as the classifier. You can can collect data train in realtime

### What you need?
first you need to install `python 2.7`

then you need to install some library. In this project I use `opencv` for image processing, `sklearn` for KNN, and `numpy` for array processing
to install them you can type this on your terminal

```
$ pip install opencv-python
$ pip install sklearn
$ pip install numpy
```


### Customization
you can edit `k` value to incrase the neigbors size.

you also can edit `face_dimen` for face image dimension

You can edit `training_count` to change the training images size

If you have more than one webcam, you can select them by change the param value on `cv2.VideoCapture` function. `0` for your first camera, `1` for your second camera and so on

You can also change the parameter to path of video file on your storage

### How to use?
open your terminal and run

```
$ python face_recognition.py
```

don't forget your computer/laptop must have webcam

Press `t` for training, just wait until it done

Press `r` to rotate the video
