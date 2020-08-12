import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

capture_face = cv2.VideoCapture(0)

while True:
    _, img = capture_face.read()
    gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_scale, 1.1,2)

    for(a,b,c,d) in faces:
        cv2.rectangle(img, (a,b), (a+c, c+d), (255, 0, 0), 2)

    cv2.imshow('img', img)

    key = cv2.waitKey(30) & 0xff

    if key == 27:
        break

capture_face.release()
