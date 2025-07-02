#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>

// 입력 이미지 크기 및 임계값 설정
const int input_width = 320;
const int input_height = 320;
const float conf_threshold = 0.3;
const float iou_threshold = 0.45;
const int target_class = 0; // 'person' 클래스만 탐지

// 탐지 결과 저장 구조체
struct Detection {
    int class_id;
    float confidence;
    cv::Rect box;
};

// IoU 계산 함수
float iou(const cv::Rect& a, const cv::Rect& b) {
    float inter = (a & b).area();
    float uni = a.area() + b.area() - inter;
    return inter / uni;
}

// Non-Maximum Suppression 수행
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

// Letterbox 전처리 함수 (비율 유지 resize + padding)
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
    // OpenVINO 모델 로딩 및 컴파일
    ov::Core core;
    auto model = core.read_model("yolo11n_openvino_model/yolo11n.xml");
    auto compiled_model = core.compile_model(model, "CPU");
    auto infer_request = compiled_model.create_infer_request();

    // 카메라 초기화 (video device 2)
    cv::VideoCapture cap(2);
    if (!cap.isOpened()) {
        std::cerr << "카메라 열기 실패" << std::endl;
        return 1;
    }

    // 카메라 해상도 설정 (필요시)
    // 기본 해상도는 640x480
    // cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    // cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    using clock = std::chrono::high_resolution_clock;
    auto start_time = clock::now();
    int frame_count = 0;

    while (true) {
        auto t1 = clock::now();
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;
        frame_count++;

        // Letterbox 전처리
        cv::Mat input_img;
        float scale;
        int pad_x, pad_y;
        letterbox(frame, input_img, scale, pad_x, pad_y);

        // BGR → RGB 변환
        cv::cvtColor(input_img, input_img, cv::COLOR_BGR2RGB);

        // 정규화
        input_img.convertTo(input_img, CV_32F, 1.0 / 255.0);

        // float 버퍼 생성 (HWC → CHW)
        std::vector<float> input_data(3 * input_height * input_width);
        int idx = 0;
        for (int c = 0; c < 3; ++c) {
            for (int i = 0; i < input_height; ++i) {
                for (int j = 0; j < input_width; ++j) {
                    input_data[idx++] = input_img.at<cv::Vec3f>(i, j)[c];
                }
            }
        }

        // Tensor 생성 후 추론 실행
        ov::Tensor input_tensor = ov::Tensor(ov::element::f32,
                                             {1, 3, input_height, input_width},
                                             input_data.data());
        infer_request.set_input_tensor(input_tensor);
        infer_request.infer();

        // 출력 Tensor에서 결과 추출
        ov::Tensor output = infer_request.get_output_tensor(0);
        const float* data = output.data<float>();
        auto shape = output.get_shape(); // [1, 25200, 85]

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

            // 원본 이미지 좌표로 복원
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

        // NMS 후 최종 탐지 결과 표시
        auto results = nms(detections);
        for (const auto& det : results) {
            cv::rectangle(frame, det.box, cv::Scalar(0, 255, 0), 2);
            std::string label = "person: " + cv::format("%.2f", det.confidence);
            cv::putText(frame, label, det.box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 255, 0}, 1);
        }

        // FPS 표시
        auto t2 = clock::now();
        double fps = 1e9 / std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
        cv::putText(frame, "FPS: " + cv::format("%.2f", fps), {10, 30},
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, {255, 255, 255}, 2);

        cv::imshow("YOLO11n OpenVINO", frame);
        if (cv::waitKey(1) == 27) break; // ESC 키로 종료
    }

    // 최종 통계 출력
    auto end_time = clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / 1000.0;
    std::cout << "총 프레임 수: " << frame_count << std::endl;
    std::cout << "총 실행 시간: " << elapsed << "초" << std::endl;
    std::cout << "평균 FPS: " << (frame_count / elapsed) << std::endl;

    return 0;
}

// compile with: g++ mainv11.cpp -o test_openvino_yolov11 -std=c++17 \
//  `pkg-config --cflags --libs opencv4` \
//  -I/home/park/openvino/src/inference/include \
//  -I/home/park/openvino/src/core/include \
//  -L/home/park/openvino/bin/aarch64/Release \
//  -lopenvino -lopencv_dnn
