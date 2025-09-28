OpenCV 환경 세팅
v4l2-ctl 유틸리티 사용을 위한 패키지 설치
sudo apt update
sudo apt install v4l2-utils
v4l2looopback 설치
sudo apt install v4l2loopback-dkms
연결된 v4l2 장치 목록 확인
v4l2-ctl --list-devices
아래와 같은 장치가 나오면 됨

unicam (platform:unicam): /dev/video0 /dev/media0

unicam: CSI-2 인터페이스와 연결된 카메라 센서를 나타냅니다. (/dev/video0)

/dev 디렉토리 확인하여 비디오 장치 노드가 생성되었는지 확인
ls /dev/video*
C++용 OpenCV 개발 라이브러리 설치
sudo apt update
sudo apt install libopencv-dev
시스템 업데이트 및 의존성 패키지 설치
# 시스템 업데이트
sudo apt-get update
sudo apt-get full-upgrade -y

# 빌드 도구 및 필수 라이브러리
sudo apt-get install -y build-essential cmake git pkg-config

# 이미지 I/O 라이브러리 (JPEG, PNG, TIFF 등)
sudo apt-get install -y libjpeg-dev libtiff-dev libpng-dev libwebp-dev

# 비디오 I/O 라이브러리 (V4L2, GStreamer, FFmpeg 등) - 핵심 부분!
sudo apt-get install -y libv4l-dev v4l-utils
sudo apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libavresample-dev

# GUI 백엔드 라이브러리 (GTK)
sudo apt-get install -y libgtk-3-dev

# 최적화 및 기타 라이브러리
sudo apt-get install -y libatlas-base-dev gfortran
sudo apt install libcamera-apps
스왑 공간 늘리기
라즈베리파이는 RAM이 부족하여 컴파일 과정에서 메모리 부족으로 멈출 수 있습니다. 일시적으로 스왑 공간을 2GB로 늘려 이 문제를 방지

# 스왑 설정 파일 열기
sudo nano /etc/dphys-swapfile
파일 내용에서 CONF_SWAPSIZE=100 부분을 찾아 CONF_SWAPSIZE=2048 으로 수정합니다.

(수정 전) CONF_SWAPSIZE=100

(수정 후) CONF_SWAPSIZE=2048

스왑 서비스 재시작
sudo /etc/init.d/dphys-swapfile restart
OpenCV 소스 코드 다운로드
홈 디렉토리에서 안정적인 최신 버전의 OpenCV와 추가 모듈(opencv_contrib) 소스 코드를 다운로드합니다. (예: 4.9.0 버전)

cd ~
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.9.0.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.9.0.zip

unzip opencv.zip
unzip opencv_contrib.zip
빌드 설정 (CMake)
cmake를 통해 어떤 기능(V4L2, GStreamer 등)을 포함하여 빌드할지 결정

# 다운로드한 opencv 디렉토리로 이동
cd ~/opencv-4.9.0/

# 빌드를 위한 별도 디렉토리 생성 및 진입
mkdir build
cd build

# CMake 설정 실행 (아래 명령어를 통째로 복사해서 터미널에 붙여넣으세요)
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib-4.9.0/modules \
    -D ENABLE_NEON=ON \
    -D WITH_V4L=ON \
    -D WITH_LIBV4L=ON \
    -D WITH_GSTREAMER=ON \
    -D WITH_GTK=ON \
    -D BUILD_EXAMPLES=ON \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D PYTHON_EXECUTABLE=/usr/bin/python3 \
    ..
WITH_V4L=ON, WITH_LIBV4L=ON: V4L2 지원을 활성화하는 가장 핵심적인 옵션입니다.
WITH_GSTREAMER=ON: GStreamer 지원도 함께 활성화합니다.
OPENCV_EXTRA_MODULES_PATH: 추가 기능(얼굴인식, SIFT 등)을 포함시킵니다.
컴파일
매우 매우 시간이 오래 걸림

make -j$(nproc)
컴파일 완료 후 시스템에 설치
sudo make install
sudo ldconfig
스왑 공간 원상 복구
sudo nano /etc/dphys-swapfile
CONF_SWAPSIZE=2048 을 다시 CONF_SWAPSIZE=100 으로 수정하고 저장 후 재시작

sudo /etc/init.d/dphys-swapfile restart
OpenVINO 빌드
OpenVINO 소스 클론
git clone --recursive https://github.com/openvinotoolkit/openvino.git
cd openvino
scons 설치
sudo apt update
sudo apt install scons
cMake 빌드
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_PYTHON=OFF \
  -DENABLE_INTEL_MYRIAD=OFF \
  -DENABLE_INTEL_GPU=OFF \
  -DENABLE_SAMPLES=ON \
  -DCMAKE_INSTALL_PREFIX=/usr/local
  
make -j4
sudo make install
정상 설치 확인
ls /usr/local/include/openvino
ls /usr/local/lib | grep openvino
.pt파일 OpenVino 형식으로 변환
변환은 PC환경에서 아나콘다 프롬프트 가상환경으로 진행한다

conda create -n model_convert python=3.8 -y
conda activate model_convert
pip install torch torchvision torchaudio
pip install tensorflow==2.10.0
pip install onnx==1.12.0 onnxruntime==1.13.1
pip install onnx-tf==1.9.0
pip install openvino-dev
cd C:\model
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
python export.py --weights ../yolov5n.pt --include onnx --img 320 --opset 12
mo --input_model yolov5n.onnx --output_dir openvino_model

mo --input_model yolo11n.onnx --output_dir openvino_model
코드 동작하기
/dev/video2 가상 장치 생성
sudo modprobe v4l2loopback video_nr=2 card_label="VirtualCam" exclusive_caps=1
/dev/video2가 생성됩니다.
video_nr=2는 원하는 번호로 바꿔도 됩니다.
가상 장치 확인
ls /dev/video2
v4l2-ctl --list-devices
가상 장치 제거
sudo modprobe -r v4l2loopback
가상 장치에 pi camera 영상 전송
720p30

libcamera-vid -t 0 --width 1280 --height 720 --framerate 30 --codec yuv420 --inline --nopreview -o - | \
ffmpeg -f rawvideo -pix_fmt yuv420p -s 1280x720 -r 30 -i - -f v4l2 /dev/video2
480p90

libcamera-vid -t 0 --width 640 --height 480 --framerate 90 --codec yuv420 --inline --nopreview -o - | \
ffmpeg -f rawvideo -pix_fmt yuv420p -s 640x480 -r 90 -i - -f v4l2 /dev/video2
이 명령어는 아래와 같은 흐름으로 작동

전체 구조
PiCamera (libcamera-vid) → YUV420 raw 데이터 → 표준 출력(stdout)
↓
ffmpeg ← 표준 입력(stdin) ← YUV420 입력
↓
/dev/video2 (v4l2loopback 가상 카메라 장치)
libcamera-vid로 영상 스트리밍해서 YUV420 포맷으로 ffmpeg에 데이터 전달

ffmpeg에서 V4L2 가상 장치 /dev/video2로 영상 전달

영상 데이터는 최초에는 raw상태로 받아진게 yuv420으로 인코딩 되고 ffmpeg에서는 딱히 데이터 인코딩 없이 바로 v4l2로 전송