#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <nlohmann/json.hpp>
#include <curl/curl.h>
#include <sstream>
#include <iomanip>
#include <thread>

using json = nlohmann::json;

const int input_width = 320;
const int input_height = 320;
const float conf_threshold = 0.3;
const float iou_threshold = 0.45;
const int target_class = 0; // class 0: person

// 메타데이터 전송 설정
const std::string METADATA_SERVER_URL = "http://192.168.0.4:8080/metadata";

struct Detection {
    int class_id;
    float confidence;
    cv::Rect box;
    std::chrono::system_clock::time_point timestamp;
};

// HTTP POST를 위한 콜백 함수
static size_t WriteCallback(void *contents, size_t size, size_t nmemb, std::string *data) {
    data->append((char*)contents, size * nmemb);
    return size * nmemb;
}

float iou(const cv::Rect& a, const cv::Rect& b) {
    float inter = (a & b).area();
    float uni = a.area() + b.area() - inter;
    return inter / uni;
}

// JSON 메타데이터 생성 함수 (nlohmann/json 버전)
json createMetadata(const std::vector<Detection>& detections, int frame_width, int frame_height, double fps) {
    json metadata;
    
    // 현재 시간을 ISO 8601 형식으로 변환
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << ms.count() << 'Z';
    
    metadata["timestamp"] = ss.str();
    metadata["frame_width"] = frame_width;
    metadata["frame_height"] = frame_height;
    metadata["fps"] = fps;
    metadata["detection_count"] = detections.size();
    
    // nlohmann/json에서는 배열 생성이 매우 간단
    metadata["objects"] = json::array();
    
    for (const auto& det : detections) {
        json obj = {
            {"class_id", det.class_id},
            {"class_name", "person"},  // target_class가 0(person)이므로
            {"confidence", det.confidence},
            {"bbox", {
                {"x", det.box.x},
                {"y", det.box.y},
                {"width", det.box.width},
                {"height", det.box.height}
            }},
            {"center", {
                {"x", det.box.x + det.box.width / 2},
                {"y", det.box.y + det.box.height / 2}
            }}
        };
        
        metadata["objects"].push_back(obj);
    }
    
    return metadata;
}

// HTTP POST로 메타데이터 전송 (nlohmann/json 버전)
bool sendMetadata(const json& metadata) {
    CURL *curl;
    CURLcode res;
    std::string response;
    bool success = false;
    
    curl = curl_easy_init();
    if(curl) {
        std::string json_string = metadata.dump();  // 매우 간단!
        
        struct curl_slist *headers = NULL;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        
        curl_easy_setopt(curl, CURLOPT_URL, METADATA_SERVER_URL.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_string.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L); // 5초 타임아웃
        
        res = curl_easy_perform(curl);
        
        if(res == CURLE_OK) {
            long response_code;
            curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
            if(response_code >= 200 && response_code < 300) {
                success = true;
            }
        }
        
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
    }
    
    return success;
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
        auto detection_time = std::chrono::system_clock::now();
        
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

            detections.push_back({class_id, conf, cv::Rect(x, y, box_w, box_h), detection_time});
        }

        auto results = nms(detections);
        
        // 실시간 FPS 계산
        auto t2 = clock::now();
        double fps = 1e9 / std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
        
        // JSON 메타데이터 생성 및 전송 (감지된 객체가 있을 때만)
        if (!results.empty()) {
            json metadata = createMetadata(results, frame.cols, frame.rows, fps);
            
            // 메타데이터를 콘솔에 출력 (디버깅용) - nlohmann/json은 indent 파라미터로 예쁘게 출력
            std::cout << "Sending metadata: " << metadata.dump(2) << std::endl;
            
            // 메타데이터 전송 (별도 스레드에서 실행하여 메인 루프 블로킹 방지)
            std::thread([metadata]() {
                if (sendMetadata(metadata)) {
                    std::cout << "Metadata sent successfully" << std::endl;
                } else {
                    std::cout << "Failed to send metadata" << std::endl;
                }
            }).detach();
        }
        
        // 화면에 감지 결과 그리기
        for (const auto& det : results) {
            cv::rectangle(frame, det.box, cv::Scalar(0, 255, 0), 2);
            std::string label = "person: " + cv::format("%.2f", det.confidence);
            cv::putText(frame, label, det.box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 255, 0}, 1);
        }

        // 실시간 FPS 표시
        cv::putText(frame, "FPS: " + cv::format("%.2f", fps), {10, 30},
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, {255, 255, 255}, 2);
        
        // 감지된 객체 수 표시
        cv::putText(frame, "Objects: " + std::to_string(results.size()), {10, 60},
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, {255, 255, 255}, 2);

        cv::imshow("YOLOv5n OpenVINO (nlohmann/json)", frame);
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
compile with nlohmann/json: 
g++ -std=c++17 -O2 `pkg-config --cflags opencv4` mainv5_nlohmann.cpp -o test_openvino_yolov5_nlohmann \
    `pkg-config --libs opencv4` \
    -I/opt/openvino/runtime/include \
    -L/opt/openvino/runtime/lib/aarch64 \
    -lopenvino -lcurl -pthread

Note: nlohmann/json is header-only library, no linking required!
*/
