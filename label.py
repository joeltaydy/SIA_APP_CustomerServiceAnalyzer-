import cv2
import label_image
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

size = 4


classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
cap = cv2.VideoCapture(0) 

while True:
    (ret, im) = cap.read()
    im=cv2.flip(im,1,0)

    mini = cv2.resize(im, (int(im.shape[1]/size), int(im.shape[0]/size)))

    # detect MultiScale / faces 
    faces = classifier.detectMultiScale(mini)

    
    for f in faces:
        (x, y, w, h) = [v * size for v in f] #Scale the shapesize backup
        cv2.rectangle(im, (x,y), (x+w,y+h), (0,255,0), 4)
        
        cropped_face = im[y:y+h, x:x+w] # Draw rectangles around each face

        faceFile = "test.jpg" #Saving the current image from the webcam for testing.
        cv2.imwrite(faceFile, cropped_face)
        
        text = label_image.main(faceFile)# Gets result from the label_image file, i.e. customer is satisfied or not
        text = text.title()
        font = cv2.FONT_HERSHEY_TRIPLEX
        cv2.putText(im, text,(x+w,y), font, 1, (0,0,255), 2)

    # Show the image
    cv2.imshow('frame',  im)
    key = cv2.waitKey(15)
    # if Esc key is press then break out of the loop 
    if key == 27: #Esc key
        break
cap.release()
cv2.destroyAllWindows()
