# Compiler
CXX = g++
CXXFLAGS = -std=c++17 -O2

# OpenCV
PKG_CFLAGS = `pkg-config --cflags opencv4`
PKG_LIBS = `pkg-config --libs opencv4`

# OpenVINO
OPENVINO_INC = -I/opt/openvino/runtime/include
OPENVINO_LIB = -L/opt/openvino/runtime/lib/aarch64 -lopenvino

# Targets
TARGETS = test_openvino_yolov5 test_openvino_yolov11

all: $(TARGETS)

test_openvino_yolov5: mainv5.cpp
	$(CXX) $(CXXFLAGS) $(PKG_CFLAGS) mainv5.cpp -o test_openvino_yolov5 $(PKG_LIBS) $(OPENVINO_INC) $(OPENVINO_LIB)

test_openvino_yolov11: mainv11.cpp
	$(CXX) $(CXXFLAGS) $(PKG_CFLAGS) mainv11.cpp -o test_openvino_yolov11 $(PKG_LIBS) $(OPENVINO_INC) $(OPENVINO_LIB)

clean:
	rm -f $(TARGETS)
