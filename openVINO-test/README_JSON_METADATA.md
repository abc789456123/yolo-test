# YOLOv5 OpenVINO with JSON Metadata Export

이 프로젝트는 YOLOv5 모델을 OpenVINO로 실행하면서 감지된 객체의 정보를 JSON 메타데이터로 HTTP 서버에 전송하는 기능을 포함합니다.

## 기능

- YOLOv5를 사용한 실시간 사람 감지
- 감지된 객체의 정보를 JSON 형태로 구조화
- HTTP POST를 통한 메타데이터 실시간 전송
- 비동기 메타데이터 전송으로 성능 최적화

## 컴파일 방법

```bash
g++ -std=c++17 -O2 `pkg-config --cflags opencv4` mainv5.cpp -o test_openvino_yolov5_metadata \
    `pkg-config --libs opencv4` \
    -I/opt/openvino/runtime/include \
    -L/opt/openvino/runtime/lib/aarch64 \
    -lopenvino -ljsoncpp -lcurl -pthread
```

## 필요한 라이브러리

```bash
sudo apt install libjsoncpp-dev libcurl4-openssl-dev
```

## 사용 방법

### 1. 메타데이터 서버 실행

```bash
python3 metadata_server.py
```

서버가 실행되면 http://localhost:8080 에서 메타데이터를 받을 준비가 됩니다.

### 2. YOLOv5 실행

```bash
./test_openvino_yolov5_metadata
```

## JSON 메타데이터 형식

감지된 객체가 있을 때마다 다음과 같은 JSON 데이터가 전송됩니다:

```json
{
  "timestamp": "2025-09-10T10:21:52.123Z",
  "frame_width": 640,
  "frame_height": 480,
  "fps": 15.67,
  "detection_count": 2,
  "objects": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.85,
      "bbox": {
        "x": 100,
        "y": 50,
        "width": 120,
        "height": 200
      },
      "center": {
        "x": 160,
        "y": 150
      }
    }
  ]
}
```

## 메타데이터 필드 설명

- `timestamp`: 감지 시점의 ISO 8601 형식 타임스탬프
- `frame_width/height`: 입력 프레임의 해상도
- `fps`: 현재 처리 속도 (FPS)
- `detection_count`: 감지된 객체 수
- `objects`: 감지된 객체들의 배열
  - `class_id`: 클래스 ID (0: person)
  - `class_name`: 클래스 이름
  - `confidence`: 신뢰도 (0.0-1.0)
  - `bbox`: 바운딩 박스 좌표 (x, y, width, height)
  - `center`: 객체 중심점 좌표

## 설정 변경

코드 상단의 상수들을 수정하여 설정을 변경할 수 있습니다:

```cpp
const float conf_threshold = 0.3;        // 신뢰도 임계값
const float iou_threshold = 0.45;        // NMS IoU 임계값
const int target_class = 0;              // 감지할 클래스 (0: person)
const std::string METADATA_SERVER_URL = "http://localhost:8080/metadata";
```

## 서버 URL 변경

메타데이터를 다른 서버로 전송하려면 `METADATA_SERVER_URL` 상수를 수정하세요:

```cpp
const std::string METADATA_SERVER_URL = "http://your-server.com:8080/api/metadata";
```

## 주의사항

- 카메라가 연결되어 있어야 합니다 (기본적으로 카메라 인덱스 2 사용)
- YOLOv5 모델 파일이 `yolo5n_openvino_model/` 폴더에 있어야 합니다
- 메타데이터 전송이 실패해도 메인 프로그램은 계속 실행됩니다
- ESC 키를 눌러 프로그램을 종료할 수 있습니다
