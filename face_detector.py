import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('./images/sample1.jpg')

img = cv2.resize(img,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC)
img = cv2.resize(img,(1000,600),interpolation=cv2.INTER_AREA)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors= 1,
    minSize=(30, 25),
    maxSize=(40, 35),
    flags = cv2.CASCADE_SCALE_IMAGE
    )

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow('sample img', img)

cv2.waitKey(0)
cv2.destroyAllWindows()