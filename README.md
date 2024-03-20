# Face Detection and Blur
In this project, we aim to detect faces in a video containing multiple individuals and apply a blur effect to the detected faces. We utilized two pre-trained models, the Haar cascade and the YOLOv8 model, for face detection, intending to compare their performance. OpenCV's Gaussian blur function is employed to blur the detected faces effectively.
## Files:
**all-face-blur.py:**
This script implements face detection using the YOLOv8 model and applies a blur effect to all detected faces in the video.

**blur.py:**
 Here, we utilize the Haar cascade model for face detection and subsequently blur all the detected faces in the video.

**single-face-blur.py:**
In this script, we leverage the YOLOv8 model for face detection. However, instead of blurring all faces, it selectively blurs only a specific face, offering a more targeted privacy enhancement solution.

## Creating a virtual environment
```bash
python -m virtualenv blur1-env
```

## Activate Virtual evironment 
```bash
blur1-env/Scripts/Activate.ps1
```
## Installing necessary requirements
```bash
python -r requirements.txt
```
## running python file
```bash 
python blur.py
```


## YOLOv8 Face Detection:


YOLO (You Only Look Once) is a deep learning model used for object detection.
The Ultralytics library is used to load a pre-trained YOLO model specifically trained for face detection 
(yolov8n-face.pt).

The OpenCV library (cv2) is used to handle video input/output and perform image processing tasks.

YOLO model is applied to each frame of the input video to detect faces.The predict() method from the YOLO object is used to predict objects in the frame, with a confidence threshold of 0.40.

Detected face regions are obtained as bounding boxes from the prediction results.For each detected face region, Gaussian blur effect is applied to blur the face.

OpenCV's GaussianBlur() function is used to apply Gaussian blur to the face region within the bounding box.Processed frames with blurred faces are written to the output video file.Processed frames are displayed in a window named "Blurred Faces".After processing all frames, video capture and OpenCV windows are closed.
```bash 
YOLOv8 utilizes a transformed -based architecture that distinguish it from previous YOLO models.
```

## Haar Cascade Face Detection:

Haar Cascade is a machine learning-based approach used for object detection. OpenCV's built-in Haar Cascade classifier for face detection (haarcascade_frontalface_default.xml) is loaded.

Similar to the YOLO implementation, video input/output paths are specified for processing.
Each frame of the input video is processed to detect faces using the Haar Cascade classifier.OpenCV's detectMultiScale() function is used to detect faces in grayscale frames.

Detected face regions (rectangles) are obtained from the cascade classifier's output.
Gaussian blur effect is applied to each detected face region using OpenCV's GaussianBlur() function.


Processed frames with blurred faces are written to the output video file.Processed frames are displayed in a window named "Blurred Faces".After processing all frames, video captured and OpenCV windows are closed.

## yolo v/s Haar Cascade 
 YOLO is "You look only once"
for object detection , yolo divides images into multiple grid cells.It looks at each grid cell and if it founds  something important in the cell then yolo creates bounding box around that object.

Haar Cascade looks for specific patterns in an image, known as "Haar-like feautres", which are simple rectangular patterns of dark and light pixels. Haar Cascade works by sliding a window over the image and comparing the pixels values within that window to pre-defined patterns.

Haar Cascade is not as fast as YOLO and it may require more computational resources, especially with large images or complex scence. Haar Cascade struggles, when it need to  detect  the objects having different sizes and orientation.

## Output
In this problem, when detecting the faces of people in the input video YOLOv8 performs better than Haar Cascade. After face detection, Gaussian Blur effect is used to blur the faces.
##

If instead of blurring all faces, we had to blur only one specific person. we can use   
 ## Face Tracking
Rather than indiscriminately blurring all detected faces,
```bash 
python single-face-blur.py
```
 file selectively blurs only the fourth detected face in each frame. This is achieved by maintaining a list of tracked faces'coordinates and applying the blur effect only to the face identified as the fourth detected face. If less than four faces are detected, it continues searching until it finds and blurs the fourth face. This approach ensures precise and targeted blurring of a specific face throughout the video sequence, offering a more controlled and customizable privacy enhancement solution.



## Refrences

[https://github.com/akanametov/yolov8-face](https://github.com/akanametov/yolov8-face)

[https://www.youtube.com/watch?v=ugI8E5GzKyM&t=589s)](https://www.youtube.com/watch?v=ugI8E5GzKyM&t=589s)