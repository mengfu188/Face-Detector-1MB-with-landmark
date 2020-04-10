#include <stdio.h>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>

#include "FaceDetector.h"

using namespace std;

int main(int argc, char** argv)
{

    string imgPath;
    if  (argc = 1)
    {
        imgPath = "../sample.jpg";
    }
    else if (argc = 2)
    {
        imgPath = argv[1];
    }
    string param = "../model/faceDetector.param";
    string bin = "../model/faceDetector.bin";
    const int max_side = 320;

    // slim or RFB
    Detector detector(param, bin, false);
    // retinaface
    // Detector detector(param, bin, true);
    Timer timer;
    for	(int i = 0; i < 10; i++){


        cv::Mat img = cv::imread(imgPath.c_str());

        // scale
//        float long_side = std::max(img.cols, img.rows);
//        float scale = max_side/long_side;
//        cv::Mat img_scale;
//        cv::Size size = cv::Size(img.cols*scale, img.rows*scale);
//        cv::resize(img, img_scale, cv::Size(img.cols*scale, img.rows*scale));

        if (img.empty())
        {
            fprintf(stderr, "cv::imread %s failed\n", imgPath.c_str());
            return -1;
        }
        std::vector<bbox> boxes;

        timer.tic();

        detector.Detect(img, boxes);
        timer.toc("----total timer:");

        // draw image
        for (int j = 0; j < boxes.size(); ++j) {
            cv::Rect rect(boxes[j].x1, boxes[j].y1, boxes[j].x2 - boxes[j].x1, boxes[j].y2 - boxes[j].y1);
            cv::rectangle(img, rect, cv::Scalar(0, 0, 255), 1, 8, 0);
            char test[80];
            sprintf(test, "%f", boxes[j].s);

            cv::putText(img, test, cv::Size((boxes[j].x1), boxes[j].y1), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 255));
//            cv::putText(img, index, cv::Size((boxes[j].x1), boxes[j].y1 + 10), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255, 255, 0));
            cv::circle(img, cv::Point(boxes[j].point[0]._x , boxes[j].point[0]._y ), 1, cv::Scalar(0, 0, 225), 4);
            cv::circle(img, cv::Point(boxes[j].point[1]._x , boxes[j].point[1]._y ), 1, cv::Scalar(0, 255, 225), 4);
            cv::circle(img, cv::Point(boxes[j].point[2]._x , boxes[j].point[2]._y ), 1, cv::Scalar(255, 0, 225), 4);
            cv::circle(img, cv::Point(boxes[j].point[3]._x , boxes[j].point[3]._y ), 1, cv::Scalar(0, 255, 0), 4);
            cv::circle(img, cv::Point(boxes[j].point[4]._x , boxes[j].point[4]._y ), 1, cv::Scalar(255, 0, 0), 4);
        }
        cv::imwrite("test.png", img);
    }
    return 0;
}

