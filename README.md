# OpenCV
OpenCV (Open Source Computer Vision) は、画像処理、物体検出、画像認識などの画像処理を容易に実行できるように設計されたオープンソースライブラリです．
またOpenCV は、C++、Python、Java など複数のプログラミング言語で使用可能であり，画像やビデオの読み取りと書き込み，さまざまな画像処理アルゴリズムの実装、特徴の検出と記述の実行、オブジェクトの検出と追跡の実行、高度なコンピューター ビジョン技術の適用による視覚データの分析と操作を行うことができます。また，カメラ，ジオメトリと3Dの構築を含む画像処理に役立つツールも提供されます．


## Simple image viewer
### opencv_sample_00.py
このコードは，OpenCV ライブラリを使用して画像ファイルを読み取って表示する例です．
```bash
#Linux
mkdir python_opencv
cd python_opencv
```
![image00.jpg](https://github.com/oit-droneproject/opencv/blob/main/image00.png)

'image00.jpg'をダウンロードして以下のディレクトリに保存しましょう． ~/python_opencv

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
 RGBとはRed（赤），Green（緑），Blue（青）の頭文字をとったもので，光の原色です．
 RGBは，コンピュータ画面，テレビ、デジタルカメラなどの電子機器で色を表現し，表示するために使用されるカラーモデルです．RGBモデルの各色は0から255までの値で表され，0はその色の強度がないことを表し，255は最大の強度を表します．
## BRG(Opencv)
ただし，OpencvにおいてはBGRとなっているためTelloで使用する場合は注意が必要となります．
Row (height) x Column (width) x Colour (3)
- Blue
- Green
- Red


## Simple image manipulation
### opencv_sample_01.py
以下のコードはRGBを色をそれぞれ変化させるためのコードです．出力結果とコード確認してください．

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
このコードはカメラをOpencvで使用するシンプルなコードです．出力結果とコード確認してください．
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
HSV は色相、彩度、明度の略で、画像処理やコンピュータグラフィックスでよく使用されるカラーモデルのことです．この 3 つの属性に基づいて色を表すことで，RGBとは異なる画像処理を行うことができます．

[HSV](https://en.wikipedia.org/wiki/HSL_and_HSV)
## HSV(opencv)
- hue :0 - 180
- Saturation:0 - 255
- value:0 - 255 
## Change color spaces (BGR to HSV) 
このコードは、OpenCVを使用してBGRに対応するHSVに変換するものです．出力結果とコード確認してください．

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
このコードはOpenCVを使用して，ビデオ フィード内の色ベース（アジア人の肌色）のオブジェクト検出と強調表示を行います．出力結果とコード確認してください．
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
次のコードは、クリックした場所の周囲のRGBの値とHSVの値を表示します。
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
以下のプログラムは，OpenCVライブラリを使用してカメラから特定の色（この場合は肌色）を検出し，その領域を緑色に変換するものです．出力結果とコード確認してください．
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
1. 画像座標を使用して、複数の赤いオブジェクトの中心 (x, y) とサイズ (幅、高さ) を検出して出力するプログラムを作成しましょう．
2. 画像座標で複数の赤い物体と緑の物体の中心（x, y）と大きさ（幅、高さ）を同時に検出して出力するプログラムを作成しましょう．
