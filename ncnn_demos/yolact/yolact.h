#ifndef NCNN_H_
#define NCNN_H_

#include "net.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>

class Object
{
public:
    cv::Rect_<float> rect;
    int label;
    float prob;
    std::vector<float> maskdata;
    cv::Mat mask;
};

class yolact
{
private:
    ncnn::Mat yolact;
    ncnn::Mat maskmaps;
    ncnn::Mat location;
    ncnn::Mat mask;
    ncnn::Mat confidence;

    char *input_layer, *output_layer1, *output_layer2, *output_layer3, *output_layer4;

    int input_width, input_height;
    int num_threads, keep_top_k, 
    
    float nms_threash, confidence_thresh;

    static void qsort_descent_inplace(std::vector<Object> &objects);
    static void qsort_descent_inplace(std::vector<Object> &objects, int left, int right);
    static inline float intersection_area(const Object &a, const Object &b);
    static void nms_sorted_bboxes(const std::vector<Object> &objects, std::vector<int> &picked, float num_thresh);
public:
    yolact();
    ~yolact();

    static int load_model(const char *paramPath, const char *binPath);
    static int detect_yolact();
};

#endif