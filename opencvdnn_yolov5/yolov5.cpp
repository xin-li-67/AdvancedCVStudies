#include <fstream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv::dnn;

const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.45;
const float CONFIDENCE_THRESHOLD = 0.45;

const float FONT_SCALE = 0.7;
const int FONT_FACE = cv::FONT_HERSHEY_SIMPLEX;
const int THICKNESS = 1;
cv::Scalar BLACK = cv::Scalar(0,0,0);
cv::Scalar BLUE = cv::Scalar(255, 178, 50);
cv::Scalar YELLOW = cv::Scalar(0, 255, 255);
cv::Scalar RED = cv::Scalar(0,0,255);

void draw_label(cv::Mat &input_image, string label, int left, int top) {
    int baseline;
    cv::Size label_size = cv::getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseline);
    top = max(top, label_size.height);

    cv::Point tlc = cv::Point(left, top);
    cv::Point brc = cv::Point(left + label_size.width, top + label_size.height + baseline);
    cv::rectangle(input_image, tlc, brc, BLACK, cv::FILLED);
    cv::putText(input_image, label, cv::Point(left, top + label_size.height), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS);
}

vector<cv::Mat> pre_process(cv::Mat &input_image, cv::dnn::Net &net) {
    cv::Mat blob;
    cv::dnn::blobFromImage(input_image, blob, 1./255, cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);

    net.setInput(blob);

    vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    return outputs;
}

static void post_process(cv::Mat &input_image, vector<cv::Mat> &detections, const vector<string> &class_name) {
    vector<int> class_ids;
    vector<float> confidences;
    vector<cv::Rect> boxes;

    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;
    float *data = (float *)detections[0].data;

    const int dims = 85;
    const int rows = 25200;

    for (int i = 0; i < rows; i++) {
        float conf = data[4];
        if (conf >= CONFIDENCE_THRESHOLD) {
            float *class_scores = data + 5;
            cv::Mat scores(1, class_name.size(), CV_32FC1, class_scores);
            cv::Point class_id;
            double max_class_score;
            
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            if (max_class_score > SCORE_THRESHOLD) {
                confidences.push_back(conf);
                class_ids.push_back(class_id.x);

                float cx = data[0], cy = data[1];
                float w = data[2], h = data[3];
                int left = int((cx - 0.5 * w) * x_factor);
                int top = int((cy - 0.5 * h) * y_factor);
                int wid = int(w * x_factor);
                int hei = int(h * y_factor);

                boxes.push_back(cv::Rect(left, top, wid, hei));
            }
        }

        data += 85;
    }

    vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);

    for (int i = 0; i < indices.size(); i++) {
        int idx = indices[i];
        cv::Rect box = boxes[idx];

        int left = box.x;
        int top = box.y;
        int width = box.width;
        int height = box.height;

        cv::rectangle(input_image, cv::Point(left, top), cv::Point(left + width, top + height), BLUE, 3 * THICKNESS);

        string label = cv::format("%.2f", confidences[idx]);
        label = class_name[class_ids[idx]] + ":" + label;
        draw_label(input_image, label, left, top);
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    vector<string> cls_names;
    ifstream ifs("class.txt");
    string line;

    while (getline(ifs, line)) {
        cls_names.push_back(line);
    }

    cv::Mat frame;
    frame = cv::imread(argv[1]);

    cv::dnn::Net net;
    net = cv::dnn::readNet("yolov5s.onnx");

    vector<cv::Mat> detections;
    detections = pre_process(frame, net);

    cv::Mat img = frame.clone();
    post_process(img, detections, cls_names);

    vector<double> layersTimes;
    double freq = cv::getTickFrequency() / 1000;
    double t = net.getPerfProfile(layersTimes) / freq;
    string label = cv::format("Inference time : %.2f ms", t);
    cv::putText(img, label, cv::Point(20, 40), FONT_FACE, FONT_SCALE, RED);

    cv::imshow("Output", img);
    cv::waitKey(0);

    return 0;
}