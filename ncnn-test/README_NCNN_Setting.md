## NCNN 소스코드 가져오기

```
git clone --depth=1 https://github.com/Tencent/ncnn.git
```

## NCNN 빌드
```
cmake -S ncnn -B build-rpi \
  -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=/work/toolchains/rpi-aarch64.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DNCNN_VULKAN=OFF \
  -DNCNN_OPENMP=ON \
  -DNCNN_BUILD_TOOLS=OFF \
  -DNCNN_BUILD_EXAMPLES=OFF \
  -DNCNN_BUILD_TESTS=OFF

cmake --build build-rpi -j"$(nproc)"

```