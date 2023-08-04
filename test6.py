import cv2
import torch
from detect import run

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/한동근/OneDrive - 한국폴리텍대학/바탕 화면/visual studio/python practice/yolov5/best.pt')

# Webcam Inference
webcam = cv2.VideoCapture(0)  # Use the webcam (change the argument to use a specific webcam index if multiple are available)

# Best result variables to keep track of the best detection
best_result = None
best_count = 0

while True:
    ret, frame = webcam.read()
    if not ret:
        break

    # Convert the frame from BGR to RGB
    frame_rgb = frame[:, :, ::-1]

    # Inference using YOLOv5 model
    results = model(frame_rgb, size=640)

    # Results
    results.print()

    # Check if the current result has more objects detected than the previous best result
    count = sum(len(det) for det in results.pred)
    if count > best_count:
        best_result = results
        best_count = count

    # Draw bounding boxes on the frame
    for det in results.pred[0]:
        x, y, w, h, conf, cls = det
        x1, y1, x2, y2 = int(x), int(y), int(x+w), int(y+h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{results.names[int(cls)]} {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Display the webcam frame with detections
    cv2.imshow('Webcam', frame)

    # Press 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save the best result if any objects are detected
if best_result is not None:
    with open('best_result.txt', 'w') as f:
        f.write(f"Number of objects detected: {best_count}\n")
        for det in best_result.pred[0]:
            x, y, w, h, conf, cls = det
            f.write(f"Class: {results.names[int(cls)]}, Confidence: {conf:.2f}, Bounding Box: ({x}, {y}, {w}, {h})\n")

    # Save the best result image  
    best_result.save()

else:
    print("No objects detected.")

# Release the webcam and close any OpenCV windows
webcam.release()
cv2.destroyAllWindows()
