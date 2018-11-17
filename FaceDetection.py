import cv2
import matplotlib.pyplot as plt
import time

def faceDetect(face_cascade,colored_img,scalefactor=1.1):
    img_copy=colored_img.copy(); #create duplicate 
    gray_img=cv2.cvtColor(colored_img, cv2.COLOR_BGR2GRAY) #convert to gray
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=scalefactor, minNeighbors=5); #detect multiple faces
    print('Faces found: ', len(faces))
    
    for (x, y, w, h) in faces:     
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2) #draw retangle with coordinates of face detected
    return(img_copy);


#FACE DETECTION    
test_img=cv2.imread('C:/Users/vk027/Desktop/test6.jpg')

haar_face_cascade = cv2.CascadeClassifier('C:/Users/vk027/Desktop/Face-Detection-OpenCV-master/Face-Detection-OpenCV-master/data/haarcascade_frontalface_alt.xml')

img=faceDetect(haar_face_cascade,test_img)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

