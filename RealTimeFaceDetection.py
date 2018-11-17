import cv2
import matplotlib.pyplot as plt
import time

haar_face_cascade = cv2.CascadeClassifier('C:/Users/vk027/Desktop/Face-Detection-OpenCV-master/Face-Detection-OpenCV-master/data/haarcascade_frontalface_alt.xml')

def faceDetector(colored_img,scalefactor=1.1):
    img_copy=colored_img.copy(); #create duplicate 
    gray_img=cv2.cvtColor(colored_img, cv2.COLOR_BGR2GRAY) #convert to gray
    faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=scalefactor, minNeighbors=5); #detect multiple faces
    print('Faces found: ', len(faces))
    
    for (x, y, w, h) in faces:     
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2) #draw retangle with coordinates of face detected
    return(img_copy);


# Webcam setup for Face Detection
cap = cv2.VideoCapture (0)
while True:
    ret, frame = cap.read ()
    cv2.imshow ('img', faceDetector(frame))
    if cv2.waitKey (1) == 13: #13 is the Enter Key
        break
# When everything done, release the capture
cap.release ()
cv2.destroyAllWindows ()

