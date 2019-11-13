import numpy as np
from sklearn.externals import joblib
import cv2
import matplotlib.pyplot as plt

classifier = joblib.load('digitsmodel.model')

image = cv2.imread('photo_1.jpg',0)

plt.imshow(image,cmap="gray")
#blurr the image to remove noise

blured = cv2.GaussianBlur(image,(5,5),0)

plt.imshow(blured,cmap="gray")

#we need to threshold the images which returns a bool and the image

ret,thresholdImage = cv2.threshold(blured,90,255,cv2.THRESH_BINARY_INV)
plt.imshow(thresholdImage,cmap="gray")
#contour gives the rectangles
contours,_heir = cv2.findContours(thresholdImage,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

 #rects = [cv2.boundingRect(ctrs) for ctr in contours]  or
rects = []
for ctr in contours:
    rects.append(cv2.boundingRect(ctr))
    
    
#val1 = rects[2]
#cv2.rectangle(image,(val1[0],val1[1]),(val1[0]+val1[2]+val1[3]),(0,255,0),2)
#plt.imshow(image[val1[1]:val1[1]+val1[3]+10:val1[1]+val1[3])])

    
    
    
    
for (x,y,w,h) in rects:
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),3)
    leng=int(h*1.5)
    pt1 = int(y+h//2-leng//2)
    pt2 =int(x+w//2-leng//2)
    roi = thresholdImage[pt1:pt1+leng,pt2+leng]
    roi = cv2.resize(roi,(28,28))
    plt.imshow(roi)
    nbr = classifier.predict(roi.reshape(1,784))
    cv2.putText(image,str(int(nbr[0]))),(x,y),cv2.FONT_HERSHEY_DUPLEX,2,(0,255,255),3
    
    
cv2.imshow("Resulting image with rectangular ROIs",image)
cv2.waitKey()
cv2.DestroyAllWindows()
