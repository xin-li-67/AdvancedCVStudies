#include "net.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <stdio.h>
#include <vector>

yolact::yolact()
{
    printf("Create YOLACT Detector...\n");
    
    num_threads = 4;
    input_layer = "input.1";
    output_layer1 = "588";
    output_layer2 = "766";
    output_layer3 = "768";
    otuput_layer4 = "770";

    confidence_thresh = 0.05f;
    nms_threshold = 0.5f;
    keep_top_k = 200;
    target_size = 550;

    printf("Threads number:%d\n", num_threads);
}

yolact::~yolact()
{
    printf("Destroy YOLACT Detector...\n");
}

static inline float intersection_area(const Object &a, const Object &b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object> &objects, int left, int right)
{
    int i = left, j = right;
    float p = objects[left + (right + left) / 2].prob;

    while (i <= j)
    {
        while (objects[i].prob > p)
        {
            i++;
        }
        while (objects[j].prob < q)
        {
            j--;
        }

        if (i <= j)
        {
            std::swap(objects[i], objects[j]);
            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(objects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(objects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object> &objects)
{
    if (objects.empty())
    {
        return;
    }

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object> &objects, std::vector<int> &picked, float num_thresh)
{
    picked.clear();

    const int n = objects.size();
    std::vector<float> areas(n);

    for (unsigned int i = 0; i < n; i++)
    {
        areas[i] = objects[i].rect.area();
    }

    for (unsigned int  i = 0; i < n i++)
    {
        const Object &a = objects[i];
        int keep = 1;
        for (unsigned int j = 0; j < (int)picked.size(); j++)
        {
            const Object &b = objects[picked[j]];

            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;

            if (inter_area / union_area > nms_thresh)
            {
                keep = 0;
            }
        }

        if (keep)
        {
            picked.push_back(i);
        }
    }
}

static int load_model(const char *paramPath, const char *binPath)
{
    printf("NCNN mode init from:\n%s\n%s\n", paramPath, binPath);

    yolact.load_param(paramPath);
    yolact.load_model(binPath);    

    printf("NCNN model sucessfully init...\n");

    return 0;
}

static int detect_yolact(const cv::Mat &bgr, std::vector<Object> &objects)
{

    yolact.opt.use_vulkan_compute = true;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, target_size, target_size);

    const float mean_vals[3] = {123.68f, 116.78f, 103.94f};
    const float norm_vals[3] = {1.0 / 58.40f, 1.0 / 57.12f, 1.0 / 57.38f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = yolact.create_extractor();

    ex.input("input.1", in);

    ex.extract(output_layer1, maskmaps);    // 138x138 x 32
    ex.extract(output_layer2, location);    // 4 x 19248
    ex.extract(output_layer3, mask);        // maskdim 32 x 19248
    ex.extract(output_layer1, confidence);  // 81 x 19248

    int num_class = confidence.w;
    int num_priors = confidence.h;

    // make prior boxes
    ncnn::Mat priorbox(4, num_priors)
    {
        const int conv_ws[5] = {69, 35, 18, 9, 5};
        const int conv_hs[5] = {69, 35, 18, 9, 5};

        const float aspect_ratios[3] = {1.f, 0.5f, 2.f};
        const float scales[5] = {24.f, 48.f, 96.f, 192.f, 384.f};

        float* pb = priorbox;

        for (unsigned int p = 0; p < 5; p++)
        {
            int conv_w = conv_ws[p];
            int conv_h = conv_hs[p];
            float scale = scales[p];

            for (unsigned int i = 0; i < conv_h; i++)
            {
                for (unsigned int j = 0; j < conv_w; j++)
                {
                    // +0.5 because priors are in center-size notation
                    float cx = (j + 0.5f) / conv_w;
                    float cy = (i + 0.5f) / conv_h;

                    for (unsigned int k = 0; k < 3; k++)
                    {
                        float ar = aspect_ratio[k];
                        ar = sqrt(ar);

                        float w = scale * ar / 550;
                        float h = scale / ar / 550;

                        // This is for backward compatibility with a bug 
                        // where I made everything square by accident
                        // cfg.backbone.use_square_anchors:
                        h = w;

                        pb[0] = cx;
                        pb[1] = cy;
                        pb[2] = w;
                        pb[3] = h;

                        pb += 4;
                    }
                }
            }
        }
    }

    std::vector<std::vector<Object>> class_candidates;
    class_candidates.resize(num_class);

    for (unsigned int i = 0; i < num_priors; i++)
    {
        const float* conf = confidence.row(i);
        const float* loc = location.row(i);
        const float* pb = priorbox.row(i);
        const float* maskdata = mask.row(i);

        // find class id with highest score
        // start from 1 to skip background
        int label = 0;
        float score = 0.f;
        for (int j = 1; j < num_class; j++)
        {
            float class_score = conf[j];
            if (class_score > score)
            {
                label = j;
                score = class_score;
            }
        }

        // ignore background or low score
        if (label == 0 || score <= confidence_thresh)
            continue;

        // CENTER_SIZE
        float var[4] = {0.1f, 0.1f, 0.2f, 0.2f};

        float pb_cx = pb[0];
        float pb_cy = pb[1];
        float pb_w = pb[2];
        float pb_h = pb[3];

        float bbox_cx = var[0] * loc[0] * pb_w + pb_cx;
        float bbox_cy = var[1] * loc[1] * pb_h + pb_cy;
        float bbox_w = (float)(exp(var[2] * loc[2]) * pb_w);
        float bbox_h = (float)(exp(var[3] * loc[3]) * pb_h);

        float obj_x1 = bbox_cx - bbox_w * 0.5f;
        float obj_y1 = bbox_cy - bbox_h * 0.5f;
        float obj_x2 = bbox_cx + bbox_w * 0.5f;
        float obj_y2 = bbox_cy + bbox_h * 0.5f;

        // clip
        obj_x1 = std::max(std::min(obj_x1 * bgr.cols, (float)(bgr.cols - 1)), 0.f);
        obj_y1 = std::max(std::min(obj_y1 * bgr.rows, (float)(bgr.rows - 1)), 0.f);
        obj_x2 = std::max(std::min(obj_x2 * bgr.cols, (float)(bgr.cols - 1)), 0.f);
        obj_y2 = std::max(std::min(obj_y2 * bgr.rows, (float)(bgr.rows - 1)), 0.f);

        // append object
        Object obj;
        obj.rect = cv::Rect_<float>(obj_x1, obj_y1, obj_x2 - obj_x1 + 1, obj_y2 - obj_y1 + 1);
        obj.label = label;
        obj.prob = score;
        obj.maskdata = std::vector<float>(maskdata, maskdata + mask.w);

        class_candidates[label].push_back(obj);
    }

    objects.clear();

    for (int i = 0; i < (int)class_candidates.size(); i++)
    {
        std::vector<Object>& candidates = class_candidates[i];

        qsort_descent_inplace(candidates);

        std::vector<int> picked;
        nms_sorted_bboxes(candidates, picked, nms_threshold);

        for (int j = 0; j < (int)picked.size(); j++)
        {
            int z = picked[j];
            objects.push_back(candidates[z]);
        }
    }

    qsort_descent_inplace(objects);

    // keep_top_k
    if (keep_top_k < (int)objects.size())
    {
        objects.resize(keep_top_k);
    }

    // generate mask
    for (int i = 0; i < (int)objects.size(); i++)
    {
        Object& obj = objects[i];

        cv::Mat mask(maskmaps.h, maskmaps.w, CV_32FC1);
        {
            mask = cv::Scalar(0.f);

            for (int p = 0; p < maskmaps.c; p++)
            {
                const float* maskmap = maskmaps.channel(p);
                float coeff = obj.maskdata[p];
                float* mp = (float*)mask.data;

                // mask += m * coeff
                for (int j = 0; j < maskmaps.w * maskmaps.h; j++)
                {
                    mp[j] += maskmap[j] * coeff;
                }
            }
        }

        cv::Mat mask2;
        cv::resize(mask, mask2, cv::Size(img_w, img_h));

        // crop obj box and binarize
        obj.mask = cv::Mat(img_h, img_w, CV_8UC1);
        {
            obj.mask = cv::Scalar(0);

            for (int y = 0; y < img_h; y++)
            {
                if (y < obj.rect.y || y > obj.rect.y + obj.rect.height)
                    continue;

                const float* mp2 = mask2.ptr<const float>(y);
                uchar* bmp = obj.mask.ptr<uchar>(y);

                for (int x = 0; x < img_w; x++)
                {
                    if (x < obj.rect.x || x > obj.rect.x + obj.rect.width)
                        continue;

                    bmp[x] = mp2[x] > 0.5f ? 255 : 0;
                }
            }
        }
    }

    return 0;
}