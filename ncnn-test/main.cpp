#include <opencv2/opencv.hpp>
#include <ncnn/net.h>
#include <ncnn/mat.h>
#include <vector>
#include <algorithm>

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

class YoloV5
{
public:
    YoloV5();
    ~YoloV5();
    
    int load(const std::string& modelpath, bool use_gpu = false);
    int detect(const cv::Mat& rgb, std::vector<Object>& objects, float prob_threshold = 0.25f, float nms_threshold = 0.45f);
    
private:
    ncnn::Net yolov5;
    int target_size = 320;
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

YoloV5::YoloV5()
{
    yolov5.opt.use_vulkan_compute = false;
    yolov5.opt.use_fp16_packed = false;
    yolov5.opt.use_fp16_storage = false;
    yolov5.opt.use_fp16_arithmetic = false;
    yolov5.opt.use_int8_storage = false;
    yolov5.opt.use_int8_arithmetic = false;
}

YoloV5::~YoloV5()
{
}

int YoloV5::load(const std::string& modelpath, bool use_gpu)
{
    yolov5.opt.use_vulkan_compute = use_gpu;

    int ret = yolov5.load_param((modelpath + ".param").c_str());
    if (ret != 0)
    {
        fprintf(stderr, "Failed to load param file: %s\n", (modelpath + ".param").c_str());
        return ret;
    }

    ret = yolov5.load_model((modelpath + ".bin").c_str());
    if (ret != 0)
    {
        fprintf(stderr, "Failed to load model file: %s\n", (modelpath + ".bin").c_str());
        return ret;
    }

    return 0;
}

int YoloV5::detect(const cv::Mat& bgr, std::vector<Object>& objects, float prob_threshold, float nms_threshold)
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

    ncnn::Extractor ex = yolov5.create_extractor();

        ex.input("in0", in_pad);

        ncnn::Mat out;
        ex.extract("out0", out);

        // Debug: Analyze output structure
        // Enable detailed debugging for model analysis
        printf("=== Model Output Analysis ===\n");
        printf("Dimensions: c=%d, h=%d, w=%d\n", out.c, out.h, out.w);
        printf("Total elements: %d\n", out.total());
        
        if (out.h > 0 && out.w >= 85) {
            // Print statistics for different ranges
            printf("\n=== Sample Data Analysis ===\n");
            for (int sample = 0; sample < std::min(5, out.h); sample++) {
                const float* row = out.row(sample);
                printf("Row %d: coords=[%.2f,%.2f,%.2f,%.2f] obj_conf=%.6f classes=[%.6f,%.6f,%.6f,%.6f,%.6f]\n", 
                       sample, row[0], row[1], row[2], row[3], row[4], 
                       row[5], row[6], row[7], row[8], row[9]);
            }
            
            // Check value ranges
            float min_coord = 999999, max_coord = -999999;
            float min_conf = 999999, max_conf = -999999;
            float min_cls = 999999, max_cls = -999999;
            
            for (int i = 0; i < std::min(100, out.h); i++) {
                const float* row = out.row(i);
                // Coordinates
                for (int j = 0; j < 4; j++) {
                    min_coord = std::min(min_coord, row[j]);
                    max_coord = std::max(max_coord, row[j]);
                }
                // Objectness
                min_conf = std::min(min_conf, row[4]);
                max_conf = std::max(max_conf, row[4]);
                // Classes
                for (int j = 5; j < std::min(85, out.w); j++) {
                    min_cls = std::min(min_cls, row[j]);
                    max_cls = std::max(max_cls, row[j]);
                }
            }
            
            printf("\n=== Value Ranges (first 100 samples) ===\n");
            printf("Coordinates: [%.2f, %.2f]\n", min_coord, max_coord);
            printf("Objectness: [%.6f, %.6f]\n", min_conf, max_conf);
            printf("Classes: [%.6f, %.6f]\n", min_cls, max_cls);
        }
        printf("================================\n\n");

        // out의 shape: [6300, 85] (H, W)
        int num_proposals = out.h;

        std::vector<Object> proposals;
        
        for (int i = 0; i < num_proposals; i++)
        {
            const float* row = out.row(i);
            
            // NCNN model already applies sigmoid - use raw values directly
            float obj_conf = row[4];
            if (obj_conf < prob_threshold) continue;

            // NCNN model already applies sigmoid to class scores - use raw values directly
            float max_cls_score = 0.0f;
            int class_id = -1;
            for (int c = 0; c < 80; c++) {
                if (row[5 + c] > max_cls_score) {
                    max_cls_score = row[5 + c];
                    class_id = c;
                }
            }

            // Final confidence
            float conf = obj_conf * max_cls_score;
            if (conf < prob_threshold) continue;

            // Debug: Print first few detections
            if (proposals.size() < 5) {
                printf("Detection %d: obj_conf=%.4f, cls_score=%.4f, final_conf=%.4f, class=%d\n", 
                       (int)proposals.size(), obj_conf, max_cls_score, conf, class_id);
            }

            // Use raw coordinates from model (they seem to be in pixel coordinates already)
            float cx = row[0];
            float cy = row[1];
            float w = row[2];
            float h = row[3];

            // Convert to original image coordinates
            float x0 = (cx - w / 2 - (wpad / 2)) / scale;
            float y0 = (cy - h / 2 - (hpad / 2)) / scale;
            float x1 = (cx + w / 2 - (wpad / 2)) / scale;
            float y1 = (cy + h / 2 - (hpad / 2)) / scale;

            // Clamp to image bounds
            x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
            y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
            x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
            y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

            // Skip invalid boxes and apply moderate size filtering
            if (x1 <= x0 || y1 <= y0) continue;
            if ((x1 - x0) < 15 || (y1 - y0) < 15) continue;  // Moderate minimum box size
            
            // Additional filtering for very low confidence
            if (conf < prob_threshold * 1.2f) continue;  // Moderate confidence boost required

            Object obj;
            obj.label = class_id;
            obj.prob = conf;
            obj.rect = cv::Rect((int)x0, (int)y0, (int)(x1-x0), (int)(y1-y0));
            proposals.push_back(obj);
        }    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with very strict threshold to reduce overlapping detections
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, 0.2f);  // Even stricter NMS

    int count = picked.size();

    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

    return 0;
}

// COCO class names
static const char* class_names[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};

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
    // Initialize YOLOv5
    YoloV5 yolo;
    
    // Load model
    int ret = yolo.load("ncnn-model/yolov5n_320", false);
    if (ret != 0)
    {
        fprintf(stderr, "Failed to load YOLOv5 model\n");
        return -1;
    }
    
    // Initialize camera with V4L2 backend explicitly
    cv::VideoCapture cap(2, cv::CAP_V4L2);
    if (!cap.isOpened())
    {
        fprintf(stderr, "Failed to open camera with V4L2 backend\n");
        // Try without specifying backend
        cap.open(2);
        if (!cap.isOpened()) {
            fprintf(stderr, "Failed to open camera\n");
            return -1;
        }
    }
    
    printf("Camera 2 initialized successfully\n");
    
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
            printf("Camera is working properly\n");
            break;
        }
        cv::waitKey(100);
    }
    
    printf("Press 'q' to quit\n");
    
    cv::Mat frame;
    while (true)
    {
        // Capture frame
        cap >> frame;
        if (frame.empty())
        {
            fprintf(stderr, "Failed to capture frame\n");
            break;
        }
        
        // Detect objects with moderate thresholds for balanced detection
        std::vector<Object> objects;
        yolo.detect(frame, objects, 0.5f, 0.45f);  // Moderate confidence threshold
        
        printf("Detected %zu objects\n", objects.size());  // Debug output
        
        // Draw detection results
        draw_objects(frame, objects);
        
        // Display frame
        cv::imshow("YOLOv5 Detection", frame);
        
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
