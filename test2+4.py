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

    # Display the webcam frame with detections
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

# Now run the same YOLOv5 model using the 'run' function from 'detect.py'
weights_file = 'best.pt'
source_path = 0
img_size = (640, 640)

# Check if CUDA (GPU) is available and set the device accordingly
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Run the 'run' function to get object detection results
results = run(weights=weights_file, source=source_path, imgsz=img_size, device=device)

# Print the detection results
if len(results.pred[0]) > 0:
    for det in results.pred[0]:
        x, y, w, h, conf, cls = det
        print(f"Class: {results.names[int(cls)]}, Confidence: {conf:.2f}, Bounding Box: ({x}, {y}, {w}, {h})")
else:
    print("No objects detected.")
