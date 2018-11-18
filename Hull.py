import cv2
import numpy as np
import sys

if __name__ == "__main__":
    if(len(sys.argv)) < 2:
        file_path = "test2.jpg"
    else:
        file_path = sys.argv[1]

    # read image
    src = cv2.imread(file_path, 1)
    
    # show source image
    cv2.imshow("Source", src)

    # convert image to gray scale
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    
    # binary thresholding of the image
    ret, thresh = cv2.threshold(gray,99 , 255, cv2.THRESH_BINARY)
    
    # find contours
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, \
    cv2.CHAIN_APPROX_SIMPLE)
    
    # create hull array for convexHull points
    hull = []
    
    # calculate points for each contour
    for i in range(len(contours)):
        hull.append(cv2.convexHull(contours[i], False))
    
    # create an empty black image
    drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)

    # draw contours and hull points
    for i in range(len(contours)):
        color = (255, 255, 255) # color for convex hull
        # draw convex hull
        cv2.drawContours(drawing, hull, i, color, 2, 8)

    cv2.imshow("Output", drawing)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
