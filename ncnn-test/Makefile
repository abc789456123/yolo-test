CXX = g++
CXXFLAGS = -std=c++11 -O2 -Wall
OPENCV_FLAGS = $(shell pkg-config --cflags opencv4)
OPENCV_LIBS = $(shell pkg-config --libs opencv4)
INCLUDES = -I/home/park/ncnn/include $(OPENCV_FLAGS)
LIBS = -L/home/park/ncnn/lib -Wl,-rpath,/home/park/ncnn/lib -Wl,-rpath,/usr/local/lib -lncnn $(OPENCV_LIBS) -pthread

# Source files
SOURCES = main.cpp
TARGETS = yolov4_tiny

# Default target
all: $(TARGETS)

# Build YOLOv4-tiny version
yolov4_tiny: main.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LIBS)

# Clean build files
clean:
	rm -f $(TARGETS) yolov4_tiny

# Install dependencies (optional)
install-deps:
	@echo "Installing OpenCV and NCNN dependencies..."
	sudo apt update
	sudo apt install -y libopencv-dev libncnn-dev

# Help target
help:
	@echo "Available targets:"
	@echo "  all          - Build YOLOv4-tiny executable"
	@echo "  yolov4_tiny  - Build YOLOv4-tiny version"
	@echo "  clean        - Remove built executables"
	@echo "  install-deps - Install required dependencies"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Usage: make yolov4_tiny"

.PHONY: all clean install-deps help