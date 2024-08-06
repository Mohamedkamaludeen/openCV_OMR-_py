import cv2
import numpy as np
import utlis

########################################################################
webCamFeed = False  # Set to False since we're not using webcam feed in this version
pathImage = "1.jpg"
cap = cv2.VideoCapture(1)
cap.set(10, 160)
heightImg = 700
widthImg = 700
questions = 5
choices = 5
ans = [1, 2, 0, 2, 4]
########################################################################

count = 10

img = cv2.imread(pathImage)
img = cv2.resize(img, (widthImg, heightImg))  # RESIZE IMAGE
imgFinal = img.copy()
imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)  # CREATE A BLANK IMAGE FOR TESTING DEBUGGING IF REQUIRED
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # CONVERT IMAGE TO GRAY SCALE
imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # ADD GAUSSIAN BLUR
imgCanny = cv2.Canny(imgBlur, 10, 70)  # APPLY CANNY

try:
    ## FIND ALL CONTOURS
    contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # FIND ALL CONTOURS
    rectCon = utlis.rectContour(contours)  # FILTER FOR RECTANGLE CONTOURS
    biggestPoints = utlis.getCornerPoints(rectCon[0])  # GET CORNER POINTS OF THE BIGGEST RECTANGLE
    gradePoints = utlis.getCornerPoints(rectCon[1])  # GET CORNER POINTS OF THE SECOND BIGGEST RECTANGLE

    if biggestPoints.size != 0 and gradePoints.size != 0:
        # BIGGEST RECTANGLE WARPING
        biggestPoints = utlis.reorder(biggestPoints)  # REORDER FOR WARPING
        pts1 = np.float32(biggestPoints)  # PREPARE POINTS FOR WARP
        pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])  # PREPARE POINTS FOR WARP
        matrix = cv2.getPerspectiveTransform(pts1, pts2)  # GET TRANSFORMATION MATRIX
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))  # APPLY WARP PERSPECTIVE

        # SECOND BIGGEST RECTANGLE WARPING
        gradePoints = utlis.reorder(gradePoints)  # REORDER FOR WARPING
        ptsG1 = np.float32(gradePoints)  # PREPARE POINTS FOR WARP
        ptsG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])  # PREPARE POINTS FOR WARP
        matrixG = cv2.getPerspectiveTransform(ptsG1, ptsG2)  # GET TRANSFORMATION MATRIX
        imgGradeDisplay = cv2.warpPerspective(img, matrixG, (325, 150))  # APPLY WARP PERSPECTIVE

        # APPLY THRESHOLD
        imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)  # CONVERT TO GRAYSCALE
        imgThresh = cv2.threshold(imgWarpGray, 170, 255, cv2.THRESH_BINARY_INV)[1]  # APPLY THRESHOLD AND INVERSE

        boxes = utlis.splitBoxes(imgThresh)  # GET INDIVIDUAL BOXES
        countR = 0
        countC = 0
        myPixelVal = np.zeros((questions, choices))  # TO STORE THE NON-ZERO VALUES OF EACH BOX
        for image in boxes:
            totalPixels = cv2.countNonZero(image)
            myPixelVal[countR][countC] = totalPixels
            countC += 1
            if countC == choices:
                countC = 0
                countR += 1

        # FIND THE USER ANSWERS AND PUT THEM IN A LIST
        myIndex = []
        for x in range(0, questions):
            arr = myPixelVal[x]
            myIndexVal = np.where(arr == np.amax(arr))
            myIndex.append(myIndexVal[0][0])

        # COMPARE THE VALUES TO FIND THE CORRECT ANSWERS
        grading = []
        for x in range(0, questions):
            if ans[x] == myIndex[x]:
                grading.append(1)
            else:
                grading.append(0)
        score = (sum(grading) / questions) * 100  # FINAL GRADE

        # PRINT FINAL SCORE
        print(f"Score: {int(score)}%")

except Exception as e:
    print("An error occurred:", e)
    print("Unable to process image.")