# 영상 파이프라인
### /dev/video2 가상 장치 생성

```
sudo modprobe v4l2loopback video_nr=2 card_label="VirtualCam" exclusive_caps=1
```
/dev/video2가 생성됩니다.

video_nr=2는 원하는 번호로 바꿔도 됩니다.

### 가상 장치 확인

```
ls /dev/video2
v4l2-ctl --list-devices
```
### 가상 장치 제거
```
sudo modprobe -r v4l2loopback
```

### 가상 장치에 pi camera 영상 전송
```
rpicam-vid -t 0 -n \
  --width 1280 --height 720 --framerate 30 \
  --codec yuv420 -o - | \
ffmpeg -f rawvideo -pix_fmt yuv420p -s 1280x720 -r 30 -i - \
       -f v4l2 /dev/video2
```
 