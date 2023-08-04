###객체를 최대로 찾았을때 이미지 저장

import cv2
import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/한동근/OneDrive - 한국폴리텍대학/바탕 화면/visual studio/python practice/yolov5/best.pt')

# Webcam Inference
webcam = cv2.VideoCapture(0)  # Use the webcam (change the argument to use a specific webcam index if multiple are available)

best_result = None
best_count = 0

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

    # Check if the current result has more objects detected than the previous best result
    count = sum(len(det) for det in results.pred)
    if count > best_count:
        best_result = results
        best_count = count
   
    # Display the webcam frame
    cv2.imshow('Webcam', frame)
   
    # Press 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Print the best result if any objects are detected
if best_result is not None:
    best_result.save()  # or .show()
else:
    print("No objects detected.")

# Release the webcam and close any OpenCV windows
webcam.release()
cv2.destroyAllWindows()
