#include "yolact.h"

int main()
{
    static const char* class_names[] = {
        "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
        "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", 
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", 
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", 
        "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", 
        "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", 
        "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", 
        "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", 
        "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", 
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", 
        "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    };

    yolact api;

    api.load_model("", "");

    cv::Mat img = cv::imread("");
    if (img.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", "imagepath");
        return -1;
    }

    std::vector<Object> objecst;
    api.detect_yolact(img, objects);

    return 0;
}