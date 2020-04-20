//
// Created by cmf on 20-4-20.
//

#include <stdio.h>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>
#include "mtcnn.h"
#include "FaceDetector.h"

using namespace std;

int main(int argc, char **argv) {

    string param = "../model/faceDetector.param";
    string bin = "../model/faceDetector.bin";
    const int max_side = 320;

    // slim or RFB
    Detector detector(param, bin, false);
    MtcnnDetector mtcnnDetector;


    Timer timer;
    cv::VideoCapture cap(0);
    cv::Mat img;
    ncnn::Mat ncnn_img;
    vector<FaceInfo> result;
    int margin = 44;
    for (;;) {

        cap >> img;
        ncnn_img = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_BGR, img.cols, img.rows);


        std::vector<bbox> boxes;

        timer.tic();

        detector.Detect(img, boxes);
        timer.toc("----total timer:");

        // onet fine tune
        int img_w = ncnn_img.w;
        int img_h = ncnn_img.h;
        result.clear();
        for(int i = 0; i < boxes.size(); i++){
            bbox box = boxes[i];
            FaceInfo info;
            info.score = box.s;
            info.x[0] = box.x1 - margin / 2;
            info.y[0] = box.y1 + margin / 2;
            info.x[1] = box.x2 - margin / 2;
            info.y[1] = box.y2 + margin / 2;
            for(int j = 0; j < 5; j++){
                info.landmark[j*2] =box.point[j]._x;
                info.landmark[j*2+1] = box.point[j]._y;
            }

            result.push_back(info);

        }
        vector<FaceInfo> onet_results = mtcnnDetector.Onet_Detect(ncnn_img, result);
        mtcnnDetector.refine(onet_results, img_h, img_w, false);
        mtcnnDetector.doNms(onet_results, 0.7, "min");


        // draw image for onet result
        for (auto it = onet_results.begin(); it != onet_results.end(); it++)
        {
            rectangle(img, cv::Point(it->x[0], it->y[0]), cv::Point(it->x[1], it->y[1]), cv::Scalar(0, 255, 0), 2);
            circle(img, cv::Point(it->landmark[0], it->landmark[1]), 2, cv::Scalar(0, 255, 0), 2);
            circle(img, cv::Point(it->landmark[2], it->landmark[3]), 2, cv::Scalar(0, 255, 0), 2);
            circle(img, cv::Point(it->landmark[4], it->landmark[5]), 2, cv::Scalar(0, 255, 0), 2);
            circle(img, cv::Point(it->landmark[6], it->landmark[7]), 2, cv::Scalar(0, 255, 0), 2);
            circle(img, cv::Point(it->landmark[8], it->landmark[9]), 2, cv::Scalar(0, 255, 0), 2);
        }

        // draw image for ssd result
        for (int j = 0; j < boxes.size(); ++j) {


            cv::Rect rect(boxes[j].x1, boxes[j].y1, boxes[j].x2 - boxes[j].x1, boxes[j].y2 - boxes[j].y1);
            cv::rectangle(img, rect, cv::Scalar(0, 0, 255), 1, 8, 0);
            char test[80];
            sprintf(test, "%f", boxes[j].s);
            char index[80];
            sprintf(index, "%d", j);
            cv::putText(img, test, cv::Size((boxes[j].x1), boxes[j].y1), cv::FONT_HERSHEY_COMPLEX, 0.5,
                        cv::Scalar(0, 255, 255));
            cv::putText(img, index, cv::Size((boxes[j].x1), boxes[j].y1 + 10), cv::FONT_HERSHEY_COMPLEX, 0.5,
                        cv::Scalar(255, 255, 0));
            cv::circle(img, cv::Point(boxes[j].point[0]._x, boxes[j].point[0]._y), 1, cv::Scalar(0, 0, 225), 4);
            cv::circle(img, cv::Point(boxes[j].point[1]._x, boxes[j].point[1]._y), 1, cv::Scalar(0, 255, 225), 4);
            cv::circle(img, cv::Point(boxes[j].point[2]._x, boxes[j].point[2]._y), 1, cv::Scalar(255, 0, 225), 4);
            cv::circle(img, cv::Point(boxes[j].point[3]._x, boxes[j].point[3]._y), 1, cv::Scalar(0, 255, 0), 4);
            cv::circle(img, cv::Point(boxes[j].point[4]._x, boxes[j].point[4]._y), 1, cv::Scalar(255, 0, 0), 4);
        }
        cv::imshow("show", img);
        int ret = cv::waitKey(1);
    }
    return 0;
}

