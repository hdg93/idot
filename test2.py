# main.py
import torch
from detect import run

weights_file = 'best.pt'
source_path = 0
img_size = (640, 640)

# 모델이 CUDA(GPU)를 지원한다면 CUDA 디바이스를 설정
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# run 함수를 호출하여 객체 인식 결과 얻기
results = run(weights=weights_file, source=source_path, imgsz=img_size, device=device)

# results 변수에 객체 인식 결과가 들어 있으므로, 이를 이용하여 추가적인 처리를 할 수 있습니다.
# 예를 들어, 바운딩 박스 좌표, 클래스, 신뢰도 등을 활용하여 다른 처리를 진행할 수 있습니다.
# ...
