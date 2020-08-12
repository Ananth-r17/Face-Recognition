import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import os
import cv2 #Import OpenCV

# Function for Face Detection
def face_detect(img):
    '''
    Takes parameters: Image
    Returns: Gray Scaled Image and detected Faces
    Prints total number of faces found
    '''

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_scale, 1.3, 5)
    total_faces = len(faces)
    print("Number of Faces: ",total_faces)
    return faces, gray_scale

# Giving labels for the Training Data
def labels_for_training_data(directory):
    '''
    Takes Parameters: the Directory which has the training Images
    Returns: Array of the faces found along with an array of the ids of the faces
    Prints the path of each image in training set and the id of the image
    '''
    faces=[]
    faceID=[]

    for path, subdirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith("."):
                print('Skipping system File')
                continue
            id1=os.path.basename(path)
            img_path=os.path.join(path, filename)
            print('img_path: ', img_path)
            print('Id: ',id1)
            test_img=cv2.imread(img_path)
            faces_rect, grey_img = face_detect(test_img)
            if len(faces_rect)==1:
                for x,y,w,h in faces_rect:
                    roi_grey = grey_img[y:y+w, x:x+h]
                    faces.append(roi_grey)
                    faceID.append(int(id1))
    return faces, faceID

#Training the Image Classifier
def train_classifier(faces, faceID):
    '''
    Parameters: Faces and Face ID from previous function
    Returns: The Trained Image Classifier
    '''
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    faceID = np.array(faceID)
    face_recognizer.train(faces, faceID)
    return face_recognizer

#Drawing Rectangle
def draw_rectangle(test_img, face):
    '''
    Parameters: The Image which has the face and the face which requires a rectangle around it
    '''
    (x,y,w,h) = face
    cv2.rectangle(test_img, (x, y), (x+w, y+h), (0, 255, 255), 3)

#Putting the Label
def put_text(test_img, text, x, y):
    '''
    Parameters: the testing image, text/label to be put, the position of Label i.e. above the rectangle
    '''
    org = (x, y-20)
    fontScale = 3
    color = (0, 255, 255)
    thickness = 4
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(test_img, text, org, font, fontScale, color, thickness, cv2.LINE_AA)

#Identifying Faces and Giving the Labels for the training set
faces, faceID = labels_for_training_data("C:\\Users\\P. R. RAJAGOPAL\\Desktop\\Data Science\\ML Models\\Images")

# Training the Model on all the Images for train Images and Giving the label names
face_recognizer = train_classifier(faces, faceID)
name = {0:"Leo", 1:"Obama"}

#Testing
capture_face = cv2.VideoCapture(0)
while True:
    _, test_img = capture_face.read()
    faces_detected, gray_img=face_detect(test_img)
    for faces in faces_detected:
        (x,y,w,h) = faces
        roi_gray = gray_img[y:y+h, x:x+h]
        label, confidence = face_recognizer.predict(roi_gray)
        #print('Confidence', confidence)
        #print('Label:', name[label])
        draw_rectangle(test_img, faces)
        predicted_name = name[label]
        put_text(test_img, predicted_name, x, y)
        #test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    cv2.imshow('Label Image', test_img)
        #plt.imshow(test_img)
        #plt.show()
    key = cv2.waitKey(30) & 0xff
    if key == 27:
        break

capture_face.release()
