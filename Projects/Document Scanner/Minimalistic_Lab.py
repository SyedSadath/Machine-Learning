# def reOrder(myPoints):
#     myPoints = myPoints.reshape((4, 2))
#     myNewPoints = np.zeros((4,1,2), np.int32)
#
#     added_sum = myPoints.sum(1) # Axes 1 i.e row wise
#     myNewPoints[0] = myPoints[np.argmin(added_sum)] # find the index of the minimum element
#     myNewPoints[3] = myPoints[np.argmax(added_sum)] # find the index of the maximum element
#
#     # diff = np.diff(myPoints, axis=1)
#     # myNewPoints[1] = myPoints[np.argmin(diff)]
#     # myNewPoints[2] = myPoints[np.argmax(diff)]
#
#     print('myNewPoints',myNewPoints)
