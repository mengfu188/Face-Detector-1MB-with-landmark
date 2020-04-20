//
// Created by cmf on 20-4-20.
//

#include "mtcnn.h"
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>

int main()
{
    MtcnnDetector detector = MtcnnDetector();
    cv::VideoCapture cap("/home/cmf/t.avi");
    cv::Mat img;
    ncnn::Mat ncnn_img;
    vector<FaceInfo> result;
    for (;;) {

        cap >> img;
        ncnn_img = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_BGR, img.cols, img.rows);
        result = detector.Detect(ncnn_img);
        for (auto it = result.begin(); it != result.end(); it++)
        {
            rectangle(img, cv::Point(it->x[0], it->y[0]), cv::Point(it->x[1], it->y[1]), cv::Scalar(0, 255, 0), 2);
            circle(img, cv::Point(it->landmark[0], it->landmark[1]), 2, cv::Scalar(0, 255, 0), 2);
            circle(img, cv::Point(it->landmark[2], it->landmark[3]), 2, cv::Scalar(0, 255, 0), 2);
            circle(img, cv::Point(it->landmark[4], it->landmark[5]), 2, cv::Scalar(0, 255, 0), 2);
            circle(img, cv::Point(it->landmark[6], it->landmark[7]), 2, cv::Scalar(0, 255, 0), 2);
            circle(img, cv::Point(it->landmark[8], it->landmark[9]), 2, cv::Scalar(0, 255, 0), 2);
        }


        cv::imshow("show", img);
        int ret = cv::waitKey(1);
    }
    return 0;
}