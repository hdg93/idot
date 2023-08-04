#####yolov5학습시킨모델 best.pt를 모듈로 가져와서 쓰기

import cv2
import torch
from PIL import Image

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/한동근/OneDrive - 한국폴리텍대학/바탕 화면/visual studio/python practice/yolov5/best.pt')

# Webcam Inference
webcam = cv2.VideoCapture(0)  # Use the webcam (change the argument to use a specific webcam index if multiple are available)
while True:
    ret, frame = webcam.read()
    if not ret:
        break

    # Convert the frame from BGR to RGB
    frame_rgb = frame[:, :, ::-1]

    # Inference
    results = model(frame_rgb, size=640)

    # Results
    results.print()  
    results.save()  # or .show()

    # Press 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close any OpenCV windows
webcam.release()
cv2.destroyAllWindows()
