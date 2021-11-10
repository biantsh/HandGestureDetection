import cv2 as cv
import numpy as np
from tensorflow import keras

################################################################################
# This file is for testing the hand detection without having to launch a movie #
################################################################################

model = keras.models.load_model('HandGestureModel')
cascade = cv.CascadeClassifier('C:/Users/Biant/trained_cascades/hand_detection/1500samples_second_try/23 stages.xml')

cap = cv.VideoCapture(0)

counter = 0

last_rectangles = []
last_imgCrop = []
last_prediction = 'No hand'

while True:
    success, img = cap.read()
    img = cv.resize(img, (750, 750))
    imgOutline = img.copy()

    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    imgGray = cv.equalizeHist(imgGray)

    imgCrop = np.zeros((128, 128))

    if counter % 5 == 0:
        rectangles = cascade.detectMultiScale(imgGray, minSize=(100, 100))

        areas = []
        for (x, y, w, h) in rectangles:
            cv.rectangle(imgOutline, (x, y), (x + w, y + h), (0, 255, 0), 3)
            area = w * h
            areas.append(area)
        for (x, y, w, h) in rectangles:
            if w * h == max(areas):
                coords = np.array([y - round(h / 2), y + h + round(h / 2), x - round(w / 2), x + w + round(w / 2)]).astype(np.int64)
                coords = np.clip(coords, 0, 750)

                print(coords)

                imgCrop = img[coords[0]:coords[1], coords[2]:coords[3]]
                imgCrop = cv.resize(imgCrop, (128, 128))

                prediction = np.argmax(model.predict(imgCrop.reshape(-1, 128, 128, 3)))
                print(prediction)
                cv.putText(imgOutline, f'Fingers: {prediction}', (10, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                last_prediction = prediction

        if len(rectangles) == 0:
            last_prediction = 'No hand'

        last_rectangles = rectangles
        last_imgCrop = imgCrop

        cv.imshow("Crop", imgCrop)
    else:
        for (x, y, w, h) in last_rectangles:
            cv.rectangle(imgOutline, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv.imshow("Crop", last_imgCrop)
        cv.putText(imgOutline, f'Fingers: {last_prediction}', (10, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

    cv.imshow("Cam", imgOutline)
    cv.waitKey(1)
    counter += 1
