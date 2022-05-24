#include <math.h>
#include <algorithm>

#include "yolo-fastestv2.h"

yoloFastestv2::yoloFastestv2()
{   
    printf("Creat yoloFastestv2 Detector...\n");
    numOutput = 2;
    numThreads = 4;
    numAnchor = 3;
    numCategory = 80;
    nmsThresh = 0.25;

    inputWidth = 352;
    inputHeight = 352;

    inputName = "input.1";
    outputName1 = "794"; // 22*22
    outputName2 = "796"; // 11*11

    printf("numThreads:%d\n", numThreads);
    printf("inputWidth:%d inputHeight:%d\n", inputWidth, inputHeight);

    std::vector<float> bias {12.64,19.39, 37.88,51.48, 55.71,138.31, 
                             126.91,78.23, 131.57,214.55, 279.92,258.87};

    anchor.assign(bias.begin(), bias.end());
}

yoloFastestv2::~yoloFastestv2()
{
    printf("Destroy yoloFastestv2 Detector...\n");
}

int yoloFastestv2::loadModel(const char *paramPath, const char *binPath)
{
    printf("NCNN mode init from:\n%s\n%s\n", paramPath, binPath);

    net.load_param(paramPath);
    net.load_model(binPath);    

    printf("NCNN model sucessfully init...\n");

    return 0;
}

float intersection_area(const TargetBox &a, const TargetBox &b)
{
    if (a.x1 > b.x2 || a.x2 < b.x1 || a.y1 > b.y2 || a.y2 < b.y1)
    {
        return 0.f;
    }

    float inter_width = std::min(a.x2, b.x2) - std::max(a.x1, b.x1);
    float inter_height = std::min(a.y2, b.y2) - std::max(a.y1, b.y1);

    return inter_width * inter_height;
}

bool scoreSort(TargetBox a, TargetBox b) 
{ 
    return (a.score > b.score); 
}

int yoloFastestv2::nmsHandle(std::vector<TargetBox> &tmpBoxes, std::vector<TargetBox> &dstBoxes)
{
    std::vector<int> picked;
    
    sort(tmpBoxes.begin(), tmpBoxes.end(), scoreSort);

    for (unsigned int i = 0; i < tmpBoxes.size(); i++)
    {
        int keep = 1;
        for (unsigned int j = 0; j < picked.size(); j++)
        {
            float inter_area = intersection_area(tmpBoxes[i], tmpBoxes[picked[j]]);
            float union_area = tmpBoxes[i].area() + tmpBoxes[picked[j]].area() - inter_area;
            float iou = inter_area / union_area;

            if(iou > nmsThresh && tmpBoxes[i].cate == tmpBoxes[picked[j]].cate)
            {
                keep = 0;
                break;
            }
        }

        if (keep) {
            picked.push_back(i);
        }
    }
    
    for (unsigned int i = 0; i < picked.size(); i++) {
        dstBoxes.push_back(tmpBoxes[picked[i]]);
    }

    return 0;
}

int yoloFastestv2::getCategory(const float *values, int index, int &category, float &score)
{
    float tmp = 0;
    float objScore  = values[4 * numAnchor + index];

    for (unsigned int i = 0; i < numCategory; i++) {
        float clsScore = values[4 * numAnchor + numAnchor + i];
        clsScore *= objScore;

        if(clsScore > tmp) {
            score = clsScore;
            category = i;
            tmp = clsScore;
        }
    }
    
    return 0;
}

int yoloFastestv2::predHandle(const ncnn::Mat *out, std::vector<TargetBox> &dstBoxes, const float scaleW, const float scaleH, const float thresh)
{
    for (unsigned int i = 0; i < numOutput; i++)
    {   
        int stride;
        int outW, outH, outC;

        outH = out[i].c;
        outW = out[i].h;
        outC = out[i].w;

        assert(inputHeight / outH == inputWidth / outW);
        stride = inputHeight / outH;

        for (unsigned int h = 0; h < outH; h++)
        {
            const float *values = out[i].channel(h);

            for (unsigned int w = 0; w < outW; w++)
            {
                for (unsigned int b = 0; b < numAnchor; b++)
                {                    
                    //float objScore = values[4 * numAnchor + b];
                    TargetBox tmpBox;
                    int category = -1;
                    float score = -1;

                    getCategory(values, b, category, score);

                    if (score > thresh)
                    {
                        float bcx, bcy, bw, bh;

                        bcx = ((values[b * 4 + 0] * 2. - 0.5) + w) * stride;
                        bcy = ((values[b * 4 + 1] * 2. - 0.5) + h) * stride;
                        bw = pow((values[b * 4 + 2] * 2.), 2) * anchor[(i * numAnchor * 2) + b * 2 + 0];
                        bh = pow((values[b * 4 + 3] * 2.), 2) * anchor[(i * numAnchor * 2) + b * 2 + 1];
                        
                        tmpBox.x1 = (bcx - 0.5 * bw) * scaleW;
                        tmpBox.y1 = (bcy - 0.5 * bh) * scaleH;
                        tmpBox.x2 = (bcx + 0.5 * bw) * scaleW;
                        tmpBox.y2 = (bcy + 0.5 * bh) * scaleH;
                        tmpBox.score = score;
                        tmpBox.cate = category;

                        dstBoxes.push_back(tmpBox);
                    }
                }
                values += outC;
            } 
        } 
    }
    return 0;
}

int yoloFastestv2::detection(const cv::Mat source, std::vector<TargetBox> &dstBoxes, const float thresh)
{   
    dstBoxes.clear();

    float scaleW = (float)source.cols / (float)inputWidth;
    float scaleH = (float)source.rows / (float)inputHeight;
    
    ncnn::Mat inputImg = ncnn::Mat::from_pixels_resize(source.data, ncnn::Mat::PIXEL_BGR, source.cols, source.rows, inputWidth, inputHeight); 

    const float mean_vals[3] = {0.f, 0.f, 0.f};
    const float norm_vals[3] = {1/255.f, 1/255.f, 1/255.f};
    inputImg.substract_mean_normalize(mean_vals, norm_vals);  

    ncnn::Extractor ex = net.create_extractor();
    ex.set_num_threads(numThreads);

    ex.input(inputName, inputImg);

    ncnn::Mat out[2]; 
    ex.extract(outputName1, out[0]); // 22*22
    ex.extract(outputName2, out[1]); // 11*11

    std::vector<TargetBox> tmpBoxes;

    predHandle(out, tmpBoxes, scaleW, scaleH, thresh);

    nmsHandle(tmpBoxes, dstBoxes);
    
    return 0;
}