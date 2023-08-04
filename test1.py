import torch
from detect_1 import detect_object_with_max_count

weights_file = 'best.pt'
source_path = 0
img_size = (640, 640)

# 모델이 CUDA(GPU)를 지원한다면 CUDA 디바이스를 설정
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 객체 인식 실행 및 결과 저장
best_frame, max_detection_count, names = detect_object_with_max_count(weights=weights_file, source=source_path,
                                                                       imgsz=img_size, device=device)

# 결과 출력
if best_frame is not None:
    print(f"Best frame has {max_detection_count} objects detected.")
    for det in best_frame:
        x, y, w, h, conf, cls = det
        print(f"Class: {names[int(cls)]}, Confidence: {conf:.2f}, Bounding Box: ({x}, {y}, {w}, {h})")
    # best_frame는 가장 많은 객체가 탐지된 프레임입니다.
    # 이를 원하는 방식으로 저장하거나, OpenCV 등을 사용하여 화면에 출력할 수 있습니다.
else:
    print("No objects detected.")
