#include <stdio.h>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>

#include "FaceDetector.h"
#include "optical_tracker.h"

using namespace std;

int main(int argc, char** argv)
{

    string param = "../model/faceDetector.param";
    string bin = "../model/faceDetector.bin";
    OpticalTracker tracker;
    int frame_index = 0;
    const int max_side = 320;

    // slim or RFB
    Detector detector(param, bin, false);
    // retinaface
    // Detector detector(param, bin, true);
    Timer timer;
//    cv::VideoCapture cap("/home/cmf/t.avi");
    cv::VideoCapture cap(0);
    cv::Mat img, old_img, new_img, _img;
    cv::Rect track_ret;
    ncnn::Mat img_n;
    int skip = 10;
    for	(;;){

//        cv::Mat img = cv::imread(imgPath.c_str());
        cap >> img;
        _img = img.clone();


        std::vector<bbox> boxes;
        timer.tic();
//        cv::resize(img, img, cv::Size(320, 160));
//        if (frame_index % skip == 0)
            detector.Detect(img, boxes);
//        img_n = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR, img.cols, img.rows, img.cols, img.rows);
        timer.toc("----total timer:");
        if(frame_index == 0) {
            tracker.init(img);
            frame_index = 1;
        }else{

//            if (boxes.size() != 0){
                // 跟丢或者到了K帧重新track
                if (track_ret.area() == 0 || frame_index % skip == 0){
                    printf("track_ret.area %d \n", track_ret.area());
                    detector.Detect(img, boxes);
                    if (!boxes.empty())
                        track_ret = tracker.track(img, old_img, boxes[0].x1, boxes[0].y1, boxes[0].x2, boxes[0].y2);

                }else{
                    track_ret = tracker.track(img, old_img, track_ret.x, track_ret.y, track_ret.x + track_ret.width, track_ret.y + track_ret.height);
                }
            printf("track_ret is %d, %d, %d, %d\n", track_ret.x, track_ret.y, track_ret.x + track_ret.width, track_ret.y + track_ret.height);
//                    printf()


//            }else{
//                // TODO 处理检测不到人的情况
//            }
            frame_index += 1;
        }
        printf("%d\n", frame_index);

        cv::rectangle(img, track_ret, cv::Scalar(0, 255, 255), 1, 8, 0);

        // draw image
        for (int j = 0; j < boxes.size(); ++j) {
            cv::Rect rect(boxes[j].x1, boxes[j].y1, boxes[j].x2 - boxes[j].x1, boxes[j].y2 - boxes[j].y1);

            cv::rectangle(img, rect, cv::Scalar(0, 0, 255), 1, 8, 0);

            char test[80];
            sprintf(test, "%f", boxes[j].s);

            char index[80];
            sprintf(index, "%d", j);
            cv::putText(img, test, cv::Size((boxes[j].x1), boxes[j].y1), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 255));
            cv::putText(img, index, cv::Size((boxes[j].x1), boxes[j].y1 + 10), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255, 255, 0));

            cv::circle(img, cv::Point(boxes[j].point[0]._x , boxes[j].point[0]._y ), 1, cv::Scalar(0, 0, 225), 4);
            cv::circle(img, cv::Point(boxes[j].point[1]._x , boxes[j].point[1]._y ), 1, cv::Scalar(0, 255, 225), 4);
            cv::circle(img, cv::Point(boxes[j].point[2]._x , boxes[j].point[2]._y ), 1, cv::Scalar(255, 0, 225), 4);
            cv::circle(img, cv::Point(boxes[j].point[3]._x , boxes[j].point[3]._y ), 1, cv::Scalar(0, 255, 0), 4);
            cv::circle(img, cv::Point(boxes[j].point[4]._x , boxes[j].point[4]._y ), 1, cv::Scalar(255, 0, 0), 4);
        }
//        cv::imwrite("test.png", img);
        cv::imshow("show", img);
        int ret = cv::waitKey(1);
        if(frame_index % skip == 0)
            old_img = _img;
    }
    return 0;
}

