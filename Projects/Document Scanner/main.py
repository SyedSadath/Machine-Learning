import cv2
import numpy as np

# webcam reading
widthImg, hieghtImg = 980, 620
Brightness = 60

url = "http://25.165.92.227:8080/video"

cap = cv2.VideoCapture(url)
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

def preProcessing(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
    imgCanny = cv2.Canny(imgBlur, 200, 200)
    kernel = np.ones((5,5))
    imgDila = cv2.dilate(imgCanny, kernel, iterations=2, )  # to thicken the edges
    imgThres = cv2.erode(imgDila, kernel, iterations=1)  # to thin the edges
    # [img, imgGray, imgBlur, imgCanny, imgDila, imgThres]
    # imgStack = stackImages(0.5, [[img, imgGray, imgBlur], [imgCanny, imgDila, imgThres]])

    return imgGray, imgBlur, imgCanny, imgDila, imgThres

def getContours(img):
    biggest = np.array([])
    maxArea = 0
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # to avoid any noise
        if area>5000:
            # curve length helps us to approximate the corner of shape
            perimeter = cv2.arcLength(cnt, True)
            # corner points count
            approx = cv2.approxPolyDP(cnt, 0.02*perimeter, True)
            if area>maxArea and len(approx)==4:
                biggest = approx
                maxArea = area
    cv2.drawContours(imgContour, biggest, -1, (255,0,0), 15)
    return biggest

def reOrder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myNewPoints = np.zeros((4,1,2), np.int32)

    added_sum = myPoints.sum(1) # Axes 1 i.e row wise
    myNewPoints[0] = myPoints[np.argmin(added_sum)] # find the index of the minimum element
    myNewPoints[3] = myPoints[np.argmax(added_sum)] # find the index of the maximum element

    diff = np.diff(myPoints, axis=1)
    myNewPoints[1] = myPoints[np.argmin(diff)]
    myNewPoints[2] = myPoints[np.argmax(diff)]

    return myNewPoints

def getWrap(img, biggest):
    biggest = reOrder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0,0], [widthImg,0], [0,hieghtImg], [widthImg,hieghtImg]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgWrap = cv2.warpPerspective(img, matrix, (widthImg, hieghtImg))

    imgCrop =   imgWrap[20:imgWrap.shape[0] - 20, 20:imgWrap.shape[1] - 20]
    imgCrop = cv2.resize(imgCrop, (widthImg,hieghtImg))

    return imgCrop

while True:
    success, img = cap.read()
    img_resized = cv2.resize(img, (widthImg, hieghtImg)) # (Width, Height)
    imgBlank = np.zeros((512, 512, 3), np.uint8)
    imgBlank = cv2.resize(imgBlank, (widthImg, hieghtImg)) # (Width, Height)
    imgGray, imgBlur, imgCanny, imgDila, imgThres = preProcessing(img_resized)
    imgContour = img_resized.copy()

    biggest = getContours(imgThres) # Image Threshold

    if biggest.size != 0:
        imgWrap = getWrap(img_resized, biggest)
        # imgStack = stackImages(0.5, [ [imgGray, imgBlur, imgCanny], [imgDila, imgThres, imgContour] ])
        imgStack = stackImages(0.5, [ [imgBlur, imgCanny, imgDila], [imgThres, imgContour, imgWrap] ])

    else:
        imgStack = stackImages(0.5, [ [imgBlur, imgCanny, imgDila], [imgThres, imgThres, imgThres] ])

    cv2.imshow('spectrum', imgWrap )
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()