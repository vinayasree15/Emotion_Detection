import cv2 as cv
import numpy as np
from predict_model import predict_emotion
from predict_emotion_with_ResNet import predict_emotion1



face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
capture = cv.VideoCapture(1)

while True:
    res, frame = capture.read()
    gray_scale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    face_rec = face_cascade.detectMultiScale(gray_scale, 1.1, 5)

    for (x, y, w, h) in face_rec:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness = 4)

        faces = frame[y:y + h, x:x + w] 
        crop_img = np.expand_dims (np. expand_dims (cv.resize(faces,(64, 64)), -1), 0)
        # output = predict_emotion('face.jpeg')
        output = predict_emotion('face.jpeg')
        cv.putText(frame, output, (x - 10, y - 10), cv.FONT_HERSHEY_COMPLEX_SMALL, 3, (0, 0, 255), thickness=4)
        cv.imwrite('face.jpeg', faces) 
        cv.imshow("frame", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv.destroyAllWindows()