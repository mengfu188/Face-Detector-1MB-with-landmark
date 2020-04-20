//
// Created by cmf on 20-4-20.
//

#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstring>
#include "FaceDetector.h"

using namespace std;
//using namespace cv;

static Timer timer;

int main(int argc, char **argv) {

    string param = "../model/faceDetector.param";
    string bin = "../model/faceDetector.bin";
    string tracker_name = "KCF";
    bool is_track=false;
    bool has_face = false;
    bool is_detect=false;
    int detect_count = 0;
    int track_count = 0;
    const int max_side = 320;

    // slim or RFB
    Detector detector(param, bin, false);

    cv::Rect2d roi;
    cv::Mat frame;
    // create a tracker object
    cv::Ptr<cv::Tracker> tracker ;
    tracker = cv::TrackerKCF::create();
//    tracker = cv::TrackerBoosting::create();
//    tracker = cv::TrackerCSRT::create();

//    cv::Ptr<cv::Tracker> tracker = cv::Tracker:;




    Timer timer;
//    cv::VideoCapture cap("/home/cmf/t.avi");
    cv::VideoCapture cap(0);
    std::vector<bbox> boxes;
    bool state=false;

    cv::Mat img, _img;
    for (;;) {

        cap >> _img;
        cv::resize(_img, _img, cv::Size(0, 0), 0.2, 0.2);
        printf("width %d, height %d\n", _img.cols, _img.rows);
//        _img.resize(320);
        _img.copyTo(img);

        // 重新检测
        boxes.clear();

        if(has_face){
            timer.tic();
            state = tracker->update(img, roi); // 图片越小越快
            timer.toc("tracker update");
            if(state){
                // 跟踪成功
                track_count += 1;
//                has_face = true;

            }else{
                // 跟踪失败
                has_face = false;
                timer.tic();
                tracker->clear();
                tracker.release();
                tracker = cv::TrackerKCF::create();
                timer.toc("tracker reset");
            }

        }else{

            timer.tic();

            detector.Detect(img, boxes);
            detect_count += 1;

            timer.toc("----total timer:");

            if(boxes.size() > 0){
                // 检测成功, 跟新roi
                has_face = true;
                bbox box = boxes[0];
                roi = cv::Rect2i(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
                timer.tic();
                tracker->init(img, roi);
                timer.toc("tracker init");

            }else{
                // 检测失败
//                has_face = false;
            }
        }

        // 信息
        char info[80];
        sprintf(info, "detect count %d, track count %d, face count %d, has face %d, track state %d", detect_count, track_count, boxes.size(), has_face, state);
        cv::putText(img, info, cv::Size(0, 20), cv::FONT_HERSHEY_COMPLEX, 0.5,
                    cv::Scalar(0, 255, 255));

        if(has_face){
            cv::rectangle(img, roi, cv::Scalar(0, 255, 0), 1, 8, 0);
        }


        // draw image
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

