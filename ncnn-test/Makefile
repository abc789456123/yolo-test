CXX = g++
CXXFLAGS = -std=c++11 -O3 -Wall
INCLUDES = -I/home/park/opencv-install/include/opencv4 -I/home/park/ncnn/include
LIBS = -L/home/park/opencv-install/lib -L/home/park/ncnn/lib
LDFLAGS = -lopencv_world -lncnn -pthread

TARGET = yolo_detection
SOURCES = main.cpp

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(TARGET) $(SOURCES) $(LIBS) $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: clean
