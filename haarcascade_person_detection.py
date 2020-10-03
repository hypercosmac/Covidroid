import numpy as np
import cv2

fullbody_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
#fullbody_cascade = cv2.CascadeClassifier('pedestrian.xml')
video_src = 'people.mp4'

cap = cv2.VideoCapture(video_src)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bodies = fullbody_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in bodies:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

    cv2.imshow('img',img)
    if cv2.waitKey(33) == 27:
        break

cap.release()
cv2.destroyAllWindows()
