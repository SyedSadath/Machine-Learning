import cv2
import numpy as np

# webcam reading
###############################
widthImg, hieghtImg = 980, 620
Brightness = 60
nPlateCascade = cv2.CascadeClassifier('/home/elliot/venv/lib/python3.8/site-packages/cv2/data/haarcascade_frontalface_default.xml')
minArea = 500
color = (255,0,0)
count =0
##############################

url = "http://25.165.92.227:8080/video"
cap = cv2.VideoCapture(0)
cap.set(3, widthImg) # id_3-width
cap.set(4, hieghtImg) # id_4-Height
cap.set(10, Brightness) # id_10- Brightness

def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale,
                                         scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver

nPlateCascade = cv2.CascadeClassifier('/home/elliot/venv/lib/python3.8/site-packages/cv2/data/haarcascade_russian_plate_number.xml')

while True:
    success, img = cap.read()

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    numberPlates = nPlateCascade.detectMultiScale(imgGray, 1.1, 4)

    for (x, y, w, h) in numberPlates:
        area = w*h
        if area > minArea:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, "Number Plate", (x,y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)

            imgRoi = img[y:y+h, x:x+w]

    cv2.imshow('spectrum', img)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite('NoPlate_'+str(count)+'.jpg',imgRoi)        # saving the image
        count += 1
        cv2.rectangle(img, (0,200), (640,300), (0,225,0), cv2.FILLED)
        cv2.putText(img, "Scan Saved", (150, 265), cv2.FONT_HERSHEY_DUPLEX, 2, (0,0,255), 2)
        cv2.imshow('Result', img)
        cv2.waitKey(500)
        break
