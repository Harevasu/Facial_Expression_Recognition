# Facial_Expression_Recognition

## Description :

    This code implements real-time Facial Expression Recognition using a pre-trained convolutional neural network (CNN). It utilizes OpenCV for face detection, extracts facial features, and feeds them into the CNN model to predict emotions such as anger, disgust, fear, happiness, neutrality, sadness, and surprise. The predicted emotion labels are overlaid on the webcam feed, providing live feedback on detected emotions.

## Team Members:
   ### ● Harevasu S
   ### ● Gurumurthy S
## Pre-Requirements:

  ● Installed Python environment with OpenCV and Keras libraries.
  ● Trained model files ('facialemotionmodel.json' and 'facialemotionmodel.h5') available.
  ● Access to a webcam for real-time video input.
  
## Project Overview:


This code is a Python script for real-time facial emotion recognition using a pre-trained convolutional neural network (CNN) model. Here's a brief explanation of each part:

### Imports:

● cv2: OpenCV library for computer vision tasks.<br>
● model_from_json from Keras: to load the trained model architecture from a JSON file.<br>
● numpy: for numerical operations.<br>
  
### Loading the Model:

● The script loads a pre-trained CNN model architecture from a JSON file (facialemotionmodel.json) and its weights from an HDF5 file (facialemotionmodel.h5).
```python
with open("facialemotionmodel.json", "r") as json_file:
    model_json = json_file.read()
    model = model_from_json(model_json)
```

### Haar Cascade Classifier:

● It loads the Haar cascade classifier for detecting faces from OpenCV's data.
```python
haar_file=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
```

### Feature Extraction Function:

● extract_features() is a function to preprocess the input image before feeding it into the neural network.

```python
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0
```

### Accessing Webcam:

● It accesses the default webcam (VideoCapture(0)).
```python
webcam=cv2.VideoCapture(0)
```

### Labels:

● A dictionary labels is defined to map the output of the model to human-readable emotion labels.
```python
labels = {0 : 'angry', 1 : 'disgust', 2 : 'fear', 3 : 'happy', 4 : 'neutral', 5 : 'sad', 6 : 'surprise'}
```

### Main Loop:

● A while loop captures frames from the webcam continuously.<br>
● Inside the loop, it converts the captured frame to grayscale and detects faces using the Haar cascade classifier.
```python
gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
```
● For each detected face, it extracts the face region, resizes it to the required input size for the model (48x48 pixels), preprocesses it, and then passes it through the model.
```python
image = cv2.resize(image,(48,48))
```
● The predicted emotion label is obtained by finding the maximum probability output from the model's prediction.
```python
pred = model.predict(img)
```
● The predicted emotion label is then overlaid on the original frame using OpenCV's putText function.<br>
● The processed frame with the predicted emotion label is displayed using imshow.<br><br>
● The loop continues until the user presses the 'c' key.
```python
if cv2.waitKey(1) & 0xFF==ord("c"):
    break
```
Overall, this script continuously captures video frames from the webcam, detects faces in each frame, predicts the emotion associated with each detected face using the pre-trained CNN model, and overlays the predicted emotion label on the video feed in real-time.<br>

Overall, this script continuously captures video frames from the webcam, detects faces in each frame, predicts the emotion associated with each detected face using the pre-trained CNN model, and overlays the predicted emotion label on the video feed in real-time.



## Output Images:

<img src="https://github.com/Harevasu/Facial_Expression_Recognition/assets/147985044/9a4e8ac1-a4f4-4a44-b133-529190b79384" width="400" height="400">

<img src="https://github.com/Harevasu/Facial_Expression_Recognition/assets/147985044/463cc2f8-abaa-4669-8ee9-d0178307562f" width="400" height="400">

<img src="https://github.com/Harevasu/Facial_Expression_Recognition/assets/147985044/0a50ce2a-1221-46a2-b97f-a9c9cd9d437d" width="400" height="400">

<img src="https://github.com/Harevasu/Facial_Expression_Recognition/assets/147985044/5acd41e6-44dd-4ecb-8cd6-8f2ddf74e48b" width="400" height="400">
