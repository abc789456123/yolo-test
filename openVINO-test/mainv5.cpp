#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>

const int input_width = 320;
const int input_height = 320;
const float conf_threshold = 0.3;
const float iou_threshold = 0.45;
const int target_class = 0; // class 0: person

struct Detection {
    int class_id;
    float confidence;
    cv::Rect box;
};

float iou(const cv::Rect& a, const cv::Rect& b) {
    float inter = (a & b).area();
    float uni = a.area() + b.area() - inter;
    return inter / uni;
}

std::vector<Detection> nms(const std::vector<Detection>& dets) {
    std::vector<Detection> res;
    std::vector<bool> suppressed(dets.size(), false);
    for (size_t i = 0; i < dets.size(); ++i) {
        if (suppressed[i]) continue;
        res.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (suppressed[j]) continue;
            if (dets[i].class_id == dets[j].class_id &&
                iou(dets[i].box, dets[j].box) > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }
    return res;
}

cv::Mat letterbox(const cv::Mat& src, cv::Mat& out, float& scale, int& pad_x, int& pad_y) {
    int w = src.cols, h = src.rows;
    scale = std::min((float)input_width / w, (float)input_height / h);
    int new_w = std::round(w * scale);
    int new_h = std::round(h * scale);
    pad_x = (input_width - new_w) / 2;
    pad_y = (input_height - new_h) / 2;
    cv::resize(src, out, cv::Size(new_w, new_h));
    cv::copyMakeBorder(out, out, pad_y, input_height - new_h - pad_y,
                              pad_x, input_width - new_w - pad_x,
                              cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    return out;
}

int main() {
    ov::Core core;
    auto model = core.read_model("yolo5n_openvino_model/yolov5n.xml");
    auto compiled_model = core.compile_model(model, "CPU");
    auto infer_request = compiled_model.create_infer_request();

    cv::VideoCapture cap(2);
    if (!cap.isOpened()) {
        std::cerr << "카메라 열기 실패" << std::endl;
        return 1;
    }

    using clock = std::chrono::high_resolution_clock;
    auto start_time = clock::now();
    int frame_count = 0;

    while (true) {
        auto t1 = clock::now();

        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;
        frame_count++;

        // Letterbox
        cv::Mat input_img;
        float scale;
        int pad_x, pad_y;
        letterbox(frame, input_img, scale, pad_x, pad_y);

        input_img.convertTo(input_img, CV_32F, 1.0 / 255.0);
        
        // Manual blob creation (NCHW format)
        cv::Mat blob(1 * 3 * input_height * input_width, 1, CV_32F);
        float* blob_data = (float*)blob.ptr();
        
        // Convert HWC to CHW format
        for (int c = 0; c < 3; ++c) {
            for (int h = 0; h < input_height; ++h) {
                for (int w = 0; w < input_width; ++w) {
                    blob_data[c * input_height * input_width + h * input_width + w] = 
                        input_img.at<cv::Vec3f>(h, w)[c];
                }
            }
        }

        ov::Tensor input_tensor = ov::Tensor(ov::element::f32,
                                             {1, 3, input_height, input_width},
                                             blob.ptr<float>());
        infer_request.set_input_tensor(input_tensor);
        infer_request.infer();
        ov::Tensor output = infer_request.get_output_tensor();
        const float* data = output.data<float>();
        auto shape = output.get_shape();  // [1, 25200, 85]

        std::vector<Detection> detections;
        for (size_t i = 0; i < shape[1]; ++i) {
            const float* row = data + i * 85;
            float obj_conf = row[4];
            if (obj_conf < 0.01f) continue;

            float max_cls_score = 0.0f;
            int class_id = -1;
            for (int c = 0; c < 80; ++c) {
                if (row[5 + c] > max_cls_score) {
                    max_cls_score = row[5 + c];
                    class_id = c;
                }
            }

            float conf = obj_conf * max_cls_score;
            if (conf < conf_threshold || class_id != target_class) continue;

            float cx = row[0], cy = row[1], w = row[2], h = row[3];
            float x0 = (cx - w / 2 - pad_x) / scale;
            float y0 = (cy - h / 2 - pad_y) / scale;
            float x1 = (cx + w / 2 - pad_x) / scale;
            float y1 = (cy + h / 2 - pad_y) / scale;

            int x = std::clamp((int)x0, 0, frame.cols - 1);
            int y = std::clamp((int)y0, 0, frame.rows - 1);
            int box_w = std::min((int)(x1 - x0), frame.cols - x);
            int box_h = std::min((int)(y1 - y0), frame.rows - y);

            detections.push_back({class_id, conf, cv::Rect(x, y, box_w, box_h)});
        }

        auto results = nms(detections);
        for (const auto& det : results) {
            cv::rectangle(frame, det.box, cv::Scalar(0, 255, 0), 2);
            std::string label = "person: " + cv::format("%.2f", det.confidence);
            cv::putText(frame, label, det.box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 255, 0}, 1);
        }

        // 실시간 FPS 표시
        auto t2 = clock::now();
        double fps = 1e9 / std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
        cv::putText(frame, "FPS: " + cv::format("%.2f", fps), {10, 30},
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, {255, 255, 255}, 2);

        cv::imshow("YOLOv5n OpenVINO", frame);
        if (cv::waitKey(1) == 27) break; // ESC 종료
    }

    auto end_time = clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / 1000.0;
    std::cout << "total frame  : " << frame_count << std::endl;
    std::cout << "total runtime: " << elapsed << "초" << std::endl;
    std::cout << "avg FPS      : " << (frame_count / elapsed) << std::endl;

    return 0;
}
/*
compile with: 
g++ mainv5.cpp -o test_openvino_yolov5 -std=c++17 \
 `pkg-config --cflags --libs opencv4` \
 -I/home/park/openvino/src/inference/include \
 -I/home/park/openvino/src/core/include \
 -L/home/park/openvino/bin/aarch64/Release \
 -lopenvino -lopencv_dnn
*/ 