# Face Recognition with OpenCV
## Introduction
How does face recognition works? It is quite simple and intuitive. Take a real life example, when you meet someone first time in your life you don't recognize him, right? While he talks or shakes hands with you, you look at his face, eyes, nose, mouth, color and overall look. This is your mind learning or training for the face recognition of that person by gathering face data. Then he tells you that his name is Obama. At this point your mind knows that the face data it just learned belongs to Paulo. Now your mind is trained and ready to do face recognition on Paulo's face. Next time when you will see Paulo or his face in a picture you will immediately recognize him. This is how face recognition work. The more you will meet Obama, the more data your mind will collect about Obama and especially his face and the better you will become at recognizing him.
The coding steps for face recognition using OpenCV are same as we discussed it in real life example above.
- <b>Training Data Gathering:</b> Gather face data (face images in this case) of the persons you want to recognize
- <b>Training of Recognizer:</b> Feed that face data (and respective names of each face) to the face recognizer so that it can learn.
- <b>Recognition:</b> Feed new faces of the persons and see if the face recognizer you just trained recognizes them.

OpenCV comes equipped with built in face recognizer, all you have to do is feed it the face data. It's that simple and this how it will look once we are done coding it.

## Images

## OpenCV Face Recognizers

OpenCV has three built in face recognizers and thanks to OpenCV's clean coding, you can use any of them by just changing a single line of code. Below are the names of those face recognizers and their OpenCV calls.

1. EigenFaces Face Recognizer Recognizer - `cv2.face.createEigenFaceRecognizer()`
2. FisherFaces Face Recognizer Recognizer - `cv2.face.createFisherFaceRecognizer()`
3. Local Binary Patterns Histograms (LBPH) Face Recognizer - `cv2.face.createLBPHFaceRecognizer()`

## Import Required Modules
Before starting the actual coding we need to import the required modules for coding. So let's import them first.

- <b>cv2:</b> is OpenCV module for Python which we will use for face detection and face recognition.
- <b>os:</b> We will use this Python module to read our training directories and file names.
- <b>numpy: </b>We will use this module to convert Python lists to numpy arrays as OpenCV face recognizers accept numpy arrays.

## Working
I am using OpenCV's LBP face detector.I convert the image to grayscale because most operations in OpenCV are performed in gray scale, then I load LBP face detector using  `cv2.CascadeClassifier class`. After that I use `cv2.CascadeClassifier class'` detectMultiScale method to detect all the faces in the image. As faces returned by detectMultiScale method are actually rectangles (x, y, width, height) and not actual faces images so we have to extract face image area from the main image. So, I extract face area from gray image and return both the face image area and face rectangle. Now, I take the training images and train the classifier based on that images. Then I test this classifier on an image or using the video camera of my PC to detect the face, draw a rectangle around the face and mention the face name on the top.

## End:
Face Recognition is a fascinating idea to work on and OpenCV has made it extremely simple and easy for us to code it. It just takes a few lines of code to have a fully working face recognition application and we can switch between all three face recognizers with a single line of code change. It's that simple.
Although EigenFaces, FisherFaces and LBPH face recognizers are good but there are even better ways to perform face recognition like using Histogram of Oriented Gradients (HOGs) and Neural Networks. So the more advanced face recognition algorithms are now a days implemented using a combination of OpenCV and Machine learning

## Author
Ananth Rajagopal:
<a href>https://github.com/Ananth-r17</a>
