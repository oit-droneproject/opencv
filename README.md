# OpenCV
OpenCV (Open Source Computer Vision) is an open-source library designed to facilitate computer vision tasks, such as image processing, object detection, and image recognition. It provides a collection of programming functions and algorithms that enable developers to perform various vision-related operations.

OpenCV supports multiple programming languages, including C++, Python, Java, and more, making it accessible to a wide range of developers. It offers a comprehensive set of functionalities for image and video processing, including image filtering, feature extraction, object tracking, camera calibration, machine learning, and more.

With OpenCV, developers can read and write images and videos, implement various image processing algorithms, perform feature detection and description, conduct object detection and tracking, and apply advanced computer vision techniques to analyze and manipulate visual data. It also provides tools for camera calibration, which is useful for computer vision applications involving camera geometries and 3D reconstruction.


## Simple image viewer
### opencv_sample_00.py
This code is an example of using the OpenCV library to read and display an image file. 
```bash
mkdir python_opencv
cd python_opencv
```
![image00.jpg](image/image00.jpg "image00.jpg")

Please download the following image as 'image00.jpg' into ~/python_opencv

```python
import cv2

WINNAME = "OpenCV Sample 00"
img = cv2.imread('./image00.jpg')
cv2.imshow(WINNAME, img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(img)
print(img.shape)
```
---
## RGB
 RGB stands for Red, Green, and Blue, which are the primary colors of light. RGB is a color model used to represent and display colors on electronic devices such as computer screens, televisions, and digital cameras. Each color in the RGB model is represented by a value ranging from 0 to 255, where 0 represents no intensity of that color, and 255 represents the highest intensity.
## BRG(Opencv)

Row (height) x Column (width) x Colour (3)
- Blue
- Green
- Red


## Simple image manipulation
### opencv_sample_01.py
This code is a code that outputs various blue colours

```python
import cv2
import numpy

WINNAME = "OpenCV Sample 01"
WIDTH = 640
HEIGHT = 480

def run():
    img = numpy.zeros((HEIGHT, WIDTH, 3))
    for r in range(256):
        for g in range(256):
            for b in range(256):
                img[...] = (b/255.0, g/255.0, r/255.0)
                print(img)
                cv2.imshow(WINNAME, img)
                key = cv2.waitKey(1)
                if key%256 == ord('q'):
                    return

if __name__ == '__main__':
    cv2.namedWindow(WINNAME)
    run()
```
---
## Camera capture
This code uses OpenCV to capture video feed from a camera and display it in a window.
### opencv_sample_02.py

```python
import cv2
import sys

WINNAME = "OpenCV Sample 02"
WIDTH = 640
HEIGHT = 480

if __name__ == '__main__':
    cv2.namedWindow(WINNAME)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit(1)

    while True:
        _, frame = cap.read()
        frame.resize((HEIGHT, WIDTH, 3))
        cv2.imshow(WINNAME, frame)
        key = cv2.waitKey(1)
        if key%256 == ord('q'):
            break
```
---
## HSV
HSV stands for Hue, Saturation, and Value, and it is a color model commonly used in image processing and computer graphics. It represents colors based on these three attributes, allowing for more intuitive manipulation and understanding of color.
[HSV](https://en.wikipedia.org/wiki/HSL_and_HSV)
## HSV(opencv)
- hue :0 - 180
- Saturation:0 - 255
- value:0 - 255 
## Change color spaces (BGR to HSV) 
This code snippet demonstrates how to convert a BGR color value to its corresponding HSV representation using OpenCV.

### opencv_sample_03.py

```python
import cv2
import numpy

if __name__ == '__main__':
    blue = numpy.uint8([[[255, 0, 0]]])  # BGR
    hsv_blue = cv2.cvtColor(blue, cv2.COLOR_BGR2HSV)
    print(hsv_blue)
```
---
## Detect asian skin color region
This code uses OpenCV to perform color-based object detection and highlighting in a video feed.
### opencv_sample_04.py
```sh
python opencv_sample_04.py
```
```python
import cv2
import numpy

WINNAME = "OpenCV Sample 04"
WIDTH = 640
HEIGHT = 480

lower = numpy.array([0, 48, 80], dtype="uint8")
upper = numpy.array([20, 255, 255], dtype="uint8")

if __name__ == '__main__':
    cv2.namedWindow(WINNAME)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit(1)

    while True:
        _, frame = cap.read()
        frame.resize((HEIGHT, WIDTH, 3))
        image = numpy.copy(frame)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hueMat = cv2.inRange(hsv, lower, upper)
        kernel = numpy.ones((3, 3), numpy.uint8)
        hueMat = cv2.erode(hueMat, kernel, iterations=3)
        hueMat = cv2.dilate(hueMat, kernel, iterations=6)
        hueMat = cv2.erode(hueMat, kernel, iterations=3)
        image[hueMat == 255] = (0, 255, 0)
        cv2.imshow(WINNAME, image)
        key = cv2.waitKey(1)
        if key % 256 == ord('q'):
            break
```
---
## Color Picker
The following code displays the rgp and hsv values around the clicked location.
### opencv_sample_05.py

 
 ```python
import numpy as np
import cv2

def mouse_event(event, x, y, flg, prm):
    if event == cv2.EVENT_LBUTTONDOWN:
        img = np.ones((128, 128, 3), np.uint8)
        avbgr = np.array([(np.uint8)(np.average(frame[y-2:y+2, x-2:x+2, 0])),
                          (np.uint8)(np.average(frame[y-2:y+2, x-2:x+2, 1])),
                          (np.uint8)(np.average(frame[y-2:y+2, x-2:x+2, 2]))])
        img[:, :, 0] = img[:, :, 0] * avbgr[0]
        img[:, :, 1] = img[:, :, 1] * avbgr[1]
        img[:, :, 2] = img[:, :, 2] * avbgr[2]
        cv2.imshow('average color', img)
        print('bgr: ' + str(img[1, 1, :]))
        avhsv = cv2.cvtColor(np.array([[avbgr]], np.uint8), cv2.COLOR_BGR2HSV)
        print('hsv: ' + str(avhsv[0, 0, :]))

cap = cv2.VideoCapture(0)
cv2.namedWindow('camera capture')
cv2.setMouseCallback('camera capture', mouse_event)

while True:
    ret, frame = cap.read()
    cv2.imshow('camera capture', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
 ```
---
## Detect center and the size for each contour
### python opencv_sample_06.py
```python
import cv2
import numpy
from numpy.random import randint

WINNAME = "OpenCV Sample 06"
WIDTH = 640
HEIGHT = 480

# blue color
# lower = numpy.array([110, 100, 100], dtype="uint8")
# upper = numpy.array([130, 255, 255], dtype="uint8")

# skin color
lower = numpy.array([0, 48, 80], dtype="uint8")
upper = numpy.array([20, 255, 255], dtype="uint8")

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit(1)

    while True:
        # Take each frame
        _, frame = cap.read()

        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Threshold the HSV image to get only skin colors
        hueMat = cv2.inRange(hsv, lower, upper)
        kernel = numpy.ones((5, 5), numpy.uint8)
        hueMat = cv2.erode(hueMat, kernel, iterations=3)
        hueMat = cv2.dilate(hueMat, kernel, iterations=6)
        hueMat = cv2.erode(hueMat, kernel, iterations=3)

        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame, frame, mask=hueMat)

        contours, hierarchy = cv2.findContours(hueMat, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        color = [randint(256) for _ in range(3)]
        cv2.drawContours(res, contours, -1, color, 3)  # draw all contours

        for cont in contours:
            M = cv2.moments(cont)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            print("(cx,cy)=(" + str(cx) + "," + str(cy) + ")")
            x, y, w, h = cv2.boundingRect(cont)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)

        cv2.imshow('frame', frame)
        cv2.imshow('mask', hueMat)
        cv2.imshow('res', res)

        key = cv2.waitKey(1)
        if key % 256 == ord('q'):
            break
```

## Exercises 
1. Make a program that detects and prints the center (x, y) and the size (width, height) of multiple red objects with an image coordinate.

2. Make a program that detects and prints the center (x, y) and the size (width, height) of multiple red objects and green objects at the same time with an image coordinate.
