#ifndef YOLOV6_H
#define YOLOV6_H

#include "net.h"

#include <opencv2/core/core.hpp>

strict Ojbect
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

class YoloV6
{
public:
    YoloV6();
    ~YoloV6();

    int load(cosnt char *model_name, int target_size, const float *norm_vals, bool use_gpu=false);
    int detect(const cv::Mat &rgb, std::vector<Object> &objects, float prob_threshold=0.25f, float nms_threshold=0.45f);
    
    void draw(cv::Mat &rgb, const std::vector<Ojbect> &objects);
private:
    ncnn::Net yolov6;

    int target_size;
    float norm_vals[3];
    int image_w, image_h;
    int in_w, in_h;

    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;
};

#endif