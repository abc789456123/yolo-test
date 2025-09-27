#include <opencv2/opencv.hpp>
#include <ncnn/net.h>
#include <ncnn/mat.h>
#include <vector>
#include <algorithm>

// YOLOv4-tiny class names - 이 모델의 클래스 매핑이 다를 수 있음
// 만약 클래스 1이 실제로 person이라면 아래와 같이 수정
static const char* class_names[] = {
    "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", 
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", 
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", 
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

class YoloV4Tiny
{
public:
    YoloV4Tiny();
    ~YoloV4Tiny();
    
    int load(const std::string& modelpath, bool use_gpu = false);
    int detect(const cv::Mat& rgb, std::vector<Object>& objects, float prob_threshold = 0.25f, float nms_threshold = 0.45f);
    
private:
    ncnn::Net yolov4;
    int target_size = 416;
    float mean_vals[3] = {0.f, 0.f, 0.f};
    float norm_vals[3] = {1/255.f, 1/255.f, 1/255.f};
    
    static inline float intersection_area(const Object& a, const Object& b)
    {
        cv::Rect_<float> inter = a.rect & b.rect;
        return inter.area();
    }
    
    static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
    {
        int i = left;
        int j = right;
        float p = faceobjects[(left + right) / 2].prob;

        while (i <= j)
        {
            while (faceobjects[i].prob > p)
                i++;

            while (faceobjects[j].prob < p)
                j--;

            if (i <= j)
            {
                std::swap(faceobjects[i], faceobjects[j]);
                i++;
                j--;
            }
        }

        if (left < j)
            qsort_descent_inplace(faceobjects, left, j);
        if (i < right)
            qsort_descent_inplace(faceobjects, i, right);
    }

    static void qsort_descent_inplace(std::vector<Object>& objects)
    {
        if (objects.empty())
            return;
        qsort_descent_inplace(objects, 0, objects.size() - 1);
    }

    static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
    {
        picked.clear();

        const int n = faceobjects.size();

        std::vector<float> areas(n);
        for (int i = 0; i < n; i++)
        {
            areas[i] = faceobjects[i].rect.area();
        }

        for (int i = 0; i < n; i++)
        {
            const Object& a = faceobjects[i];

            int keep = 1;
            for (int j = 0; j < (int)picked.size(); j++)
            {
                const Object& b = faceobjects[picked[j]];

                float inter_area = intersection_area(a, b);
                float union_area = areas[i] + areas[picked[j]] - inter_area;

                if (inter_area / union_area > nms_threshold)
                    keep = 0;
            }

            if (keep)
                picked.push_back(i);
        }
    }
};

YoloV4Tiny::YoloV4Tiny()
{
    yolov4.opt.use_vulkan_compute = false;
    yolov4.opt.use_fp16_packed = false;
    yolov4.opt.use_fp16_storage = false;
    yolov4.opt.use_fp16_arithmetic = false;
    yolov4.opt.use_int8_storage = false;
    yolov4.opt.use_int8_arithmetic = false;
}

YoloV4Tiny::~YoloV4Tiny()
{
}

int YoloV4Tiny::load(const std::string& modelpath, bool use_gpu)
{
    yolov4.opt.use_vulkan_compute = use_gpu;

    int ret = yolov4.load_param((modelpath + ".param").c_str());
    if (ret != 0)
    {
        fprintf(stderr, "Failed to load param file: %s\\n", (modelpath + ".param").c_str());
        return ret;
    }

    ret = yolov4.load_model((modelpath + ".bin").c_str());
    if (ret != 0)
    {
        fprintf(stderr, "Failed to load model file: %s\\n", (modelpath + ".bin").c_str());
        return ret;
    }

    return 0;
}

int YoloV4Tiny::detect(const cv::Mat& bgr, std::vector<Object>& objects, float prob_threshold, float nms_threshold)
{
    int img_w = bgr.cols;
    int img_h = bgr.rows;

    // letterbox pad to multiple of 32
    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);

    // pad to target_size rectangle
    int wpad = (w + 31) / 32 * 32 - w;
    int hpad = (h + 31) / 32 * 32 - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

    in_pad.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = yolov4.create_extractor();
    ex.input("data", in_pad);

    ncnn::Mat out;
    ex.extract("output", out);

    // YOLOv4-tiny with Yolov3DetectionOutput layer
    // The output is already post-processed by NCNN's Yolov3DetectionOutput layer
    // Format: [batch, num_detections, 6] where 6 = [class_id, confidence, x1, y1, x2, y2]
    
    printf("=== YOLOv4-tiny Output Analysis ===\\n");
    printf("Dimensions: c=%d, h=%d, w=%d\\n", out.c, out.h, out.w);
    printf("Total elements: %d\\n", out.total());
    
    objects.clear();
    
    if (out.h > 0) {
        for (int i = 0; i < out.h; i++) {
            const float* detection = out.row(i);
            
            int class_id = (int)detection[0];
            float confidence = detection[1];
            
            if (confidence < prob_threshold) continue;
            
            // Convert normalized coordinates (0-1) to pixel coordinates
            float x1 = detection[2] * img_w;
            float y1 = detection[3] * img_h;
            float x2 = detection[4] * img_w;
            float y2 = detection[5] * img_h;
            
            // Clamp to image bounds
            x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
            y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);
            x2 = std::max(std::min(x2, (float)(img_w - 1)), 0.f);
            y2 = std::max(std::min(y2, (float)(img_h - 1)), 0.f);
            
            // Skip invalid boxes
            if (x2 <= x1 || y2 <= y1) continue;
            if ((x2 - x1) < 10 || (y2 - y1) < 10) continue;
            
            Object obj;
            obj.label = class_id;
            obj.prob = confidence;
            obj.rect = cv::Rect_<float>(x1, y1, x2 - x1, y2 - y1);
            
            objects.push_back(obj);
            
            // Debug: Print first few detections with class names
            if (objects.size() <= 5) {
                const char* class_name = (class_id >= 0 && class_id < 80) ? class_names[class_id] : "unknown";
                printf("Detection %d: class=%d (%s), conf=%.4f, bbox=[%.1f,%.1f,%.1f,%.1f]\\n", 
                       (int)objects.size()-1, class_id, class_name, confidence, x1, y1, x2, y2);
            }
        }
    }
    
    printf("Found %d valid detections\\n", (int)objects.size());
    printf("=====================================\\n\\n");

    return 0;
}

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    static const cv::Scalar colors[19] = {
        cv::Scalar(54, 67, 244),
        cv::Scalar(99, 30, 233),
        cv::Scalar(176, 39, 156),
        cv::Scalar(183, 58, 103),
        cv::Scalar(181, 81, 63),
        cv::Scalar(243, 150, 33),
        cv::Scalar(244, 169, 3),
        cv::Scalar(212, 188, 0),
        cv::Scalar(136, 150, 0),
        cv::Scalar(80, 175, 76),
        cv::Scalar(74, 195, 139),
        cv::Scalar(57, 220, 205),
        cv::Scalar(59, 235, 255),
        cv::Scalar(7, 193, 255),
        cv::Scalar(0, 152, 255),
        cv::Scalar(34, 87, 255),
        cv::Scalar(72, 85, 121),
        cv::Scalar(158, 158, 158),
        cv::Scalar(139, 125, 96)
    };

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        const cv::Scalar& color = colors[obj.label % 19];
        cv::rectangle(bgr, obj.rect, color, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > bgr.cols)
            x = bgr.cols - label_size.width;

        cv::rectangle(bgr, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      color, -1);

        cv::putText(bgr, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
    }
}

int main()
{
    // Initialize YOLOv4-tiny
    YoloV4Tiny yolo;
    
    // Load model
    int ret = yolo.load("ncnn-model/yolov4-tiny", false);
    if (ret != 0)
    {
        fprintf(stderr, "Failed to load YOLOv4-tiny model\\n");
        return -1;
    }
    
    // Initialize camera with V4L2 backend explicitly
    cv::VideoCapture cap(2, cv::CAP_V4L2);
    if (!cap.isOpened())
    {
        fprintf(stderr, "Failed to open camera with V4L2 backend\\n");
        // Try without specifying backend
        cap.open(2);
        if (!cap.isOpened()) {
            fprintf(stderr, "Failed to open camera\\n");
            return -1;
        }
    }
    
    printf("Camera 2 initialized successfully\\n");
    
    // Set camera properties
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FPS, 30);
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    
    // Wait a moment for camera to warm up
    cv::Mat test_frame;
    for (int i = 0; i < 10; i++) {
        cap >> test_frame;
        if (!test_frame.empty()) {
            printf("Camera is working properly\\n");
            break;
        }
        cv::waitKey(100);
    }
    
    printf("Press 'q' to quit\\n");
    
    cv::Mat frame;
    while (true)
    {
        // Capture frame
        cap >> frame;
        if (frame.empty())
        {
            fprintf(stderr, "Failed to capture frame\\n");
            break;
        }
        
        // Detect objects with moderate thresholds for balanced detection
        std::vector<Object> objects;
        yolo.detect(frame, objects, 0.1f, 0.45f);  // Moderate confidence threshold
        
        printf("Detected %zu objects\\n", objects.size());  // Debug output
        
        // Draw detection results
        draw_objects(frame, objects);
        
        // Display frame
        cv::imshow("YOLOv4-tiny Detection", frame);
        
        // Check for exit
        char key = cv::waitKey(1);
        if (key == 'q' || key == 27) // 'q' or ESC
            break;
    }
    
    // Cleanup
    cap.release();
    cv::destroyAllWindows();
    
    return 0;
}