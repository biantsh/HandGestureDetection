import cv2 as cv
import numpy as np
from tensorflow import keras
import vlc
from pynput.keyboard import Key, Controller

keyboard = Controller()

# Loading the Haar cascade and Tensorflow models
cascade = cv.CascadeClassifier('cascade/cascade.xml')
model = keras.models.load_model('HandGestureModel')

# Setting up webcam capture
cap = cv.VideoCapture(0)

# Initializing variables
counter = 0
last_rectangles = last_imgCrop = []
last_prediction = 'No hand'
number = -1

movie = vlc.MediaPlayer("Soul.mp4")
movie.play()

# Mainloop
while True:
    success, img = cap.read()

    img_show = cv.resize(img, (250, 250))

    ## Preprocessing the image before detection
    img = cv.resize(img, (750, 750))
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imgGray = cv.equalizeHist(imgGray)

    ## Only performing inference on every 5th frame in order to increase smoothness
    if counter % 5 == 0:
        rectangles = cascade.detectMultiScale(imgGray, minSize=(100, 100))

        ## Finding all of the detected rectangles and calculating their area
        areas = []
        for (x, y, w, h) in rectangles:
            area = w * h
            areas.append(area)
        for (x, y, w, h) in rectangles:
            ## Only taking the biggest rectangle detected as that is assumed to be the hand
            if w * h == max(areas):
                coords = np.array([y - round(h / 2), y + h + round(h / 2), x - round(w / 2), x + w + round(w / 2)])
                coords = np.clip(coords, 0, 750)

                ## Cropping the image to only take the part containing the hand
                imgCrop = img[coords[0]:coords[1], coords[2]:coords[3]]
                imgCrop = cv.resize(imgCrop, (128, 128))

                ## Performing classification using the NN model to the cropped image
                prediction = np.argmax(model.predict(imgCrop.reshape(-1, 128, 128, 3)))
                if prediction == last_prediction:
                    ## The following triple-check system prevents false positives and allows the system to only pause once
                    ## when you raise your hand, as opposed to continuously pausing/unpausing. 
                    if prediction == 5 and prediction == number and prediction != last_printed_number:
                        last_printed_number = prediction
                        movie.pause()
                    elif prediction == number:
                        last_printed_number = prediction
                        if prediction == 0:
                            for i in range(2):
                                keyboard.press(Key.media_volume_down)
                                keyboard.release(Key.media_volume_down)
                        elif prediction == 1:
                            for i in range(2):
                                keyboard.press(Key.media_volume_up)
                                keyboard.release(Key.media_volume_up)
                    number = prediction
                last_prediction = prediction

        ## If no hand is detected (coded with the number 6)
        if len(rectangles) == 0:
            last_prediction = 'No hand'
            last_printed_number = 6
            number = 6

    cv.imshow("Camera", img_show)
    cv.waitKey(1)
    counter += 1
