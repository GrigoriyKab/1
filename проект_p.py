import numpy as np
import cv2
import math
cv2.namedWindow("mask")

def nothing(x):
    pass
low_hsv = (1, 82, 172)
high_hsv = (217, 157, 218)

lh, ls, lv = low_hsv
hh, hs, hv = high_hsv

cv2.createTrackbar("lh", "mask", lh, 255, nothing)
cv2.createTrackbar("ls", "mask", ls, 255, nothing)
cv2.createTrackbar("lv", "mask", lv, 255, nothing)
cv2.createTrackbar("hh", "mask", hh, 255, nothing)
cv2.createTrackbar("hs", "mask", hs, 255, nothing)
cv2.createTrackbar("hv", "mask", hv, 255, nothing)


while (True):
    
    frame = cv2.imread('C:/Users/kazak/Downloads/WhatsApp Image 2024-06-14 at 20.25.05.jpg')
    

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lh = cv2.getTrackbarPos("lh", "mask")
    ls = cv2.getTrackbarPos("ls", "mask")
    lv = cv2.getTrackbarPos("lv", "mask")
    hh = cv2.getTrackbarPos("hh", "mask")
    hs = cv2.getTrackbarPos("hs", "mask")
    hv = cv2.getTrackbarPos("hv", "mask")
    
    mask = cv2.inRange(hsv, (lh, ls, lv), (hh, hs, hv))
    print((lh, ls, lv), (hh, hs, hv))
    cv2.imshow("mask", mask)
    
    connectivity = 4
    
    output = cv2.connectedComponentsWithStats(mask, connectivity, cv2.CV_32S)
    
   
    num_labels = output[0]
    
    labels = output[1]
    
    stats = output[2]
    
    filtered = np.zeros_like(mask)
    
    for i in range(1, num_labels):
        a = stats[i, cv2.CC_STAT_AREA]
        t = stats[i, cv2.CC_STAT_TOP]
        l = stats[i, cv2.CC_STAT_LEFT]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        
    
        
        if (a >= 1000):
            filtered[np.where(labels == i)] = 255
          
            linear_size = math.sqrt(a)

            distance_by_cam = round(calibration_distance * calibration_linear_size / linear_size)
            
            cv2.rectangle(frame, (l, t), (l + w, t + h), (0, 255, 0), 2)
        
   
    
    cv2.imshow("frame", frame)
    
    cv2.imshow("filtered", filtered)
    
    key = cv2.waitKey(280) & 0xFF
    
    if (key == ord(' ')):
        break


cv2.destroyAllWindows()
cv2.waitKey(10)
