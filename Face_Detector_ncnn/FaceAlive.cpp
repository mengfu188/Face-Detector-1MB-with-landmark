//
// Created by cmf on 20-4-1.
//

#include <stdio.h>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>

#include "FaceDetector.h"


using namespace std;



static Timer timer;



void draw(cv::Mat &img, std::vector<bbox> boxes, float scale)
{
    // draw image
    for (int j = 0; j < boxes.size(); ++j) {
        cv::Rect rect(boxes[j].x1/scale, boxes[j].y1/scale, boxes[j].x2/scale - boxes[j].x1/scale, boxes[j].y2/scale - boxes[j].y1/scale);
        cv::rectangle(img, rect, cv::Scalar(0, 0, 255), 1, 8, 0);
        char test[80];
        sprintf(test, "%f", boxes[j].s);

        cv::putText(img, test, cv::Size((boxes[j].x1/scale), boxes[j].y1/scale), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 255));
        cv::circle(img, cv::Point(boxes[j].point[0]._x / scale, boxes[j].point[0]._y / scale), 1, cv::Scalar(0, 0, 225), 4);
        cv::circle(img, cv::Point(boxes[j].point[1]._x / scale, boxes[j].point[1]._y / scale), 1, cv::Scalar(0, 255, 225), 4);
        cv::circle(img, cv::Point(boxes[j].point[2]._x / scale, boxes[j].point[2]._y / scale), 1, cv::Scalar(255, 0, 225), 4);
        cv::circle(img, cv::Point(boxes[j].point[3]._x / scale, boxes[j].point[3]._y / scale), 1, cv::Scalar(0, 255, 0), 4);
        cv::circle(img, cv::Point(boxes[j].point[4]._x / scale, boxes[j].point[4]._y / scale), 1, cv::Scalar(255, 0, 0), 4);
    }
}

void cat(cv::Mat &rgb, cv::Mat &gray, cv::Mat out)
{

}

int cat(ncnn::Mat &img1, ncnn::Mat &img2, ncnn::Mat &top_blob)
{
    int dims = img1.dims;  // dims 3
    size_t elemsize = img1.elemsize;
    size_t elemsize2 = img2.elemsize;

    printf("img1 dims %d; elemsize %zu\n", dims, elemsize);

    int h = img1.h;
    int w = img1.w;

    int top_channels = 4;

    top_blob.create(w, h, top_channels, elemsize);
    if(top_blob.empty())
        return -100;

    size_t size = img1.cstep * 3;
    const unsigned char* ptr = img1;
    unsigned char* outptr = top_blob.channel(0);
    memcpy(outptr, ptr, size * elemsize);

    size_t size2 = img2.cstep;
    const unsigned  char* ptr2 = img2;
    unsigned  char * outptr2 = top_blob.channel(3);
    memcpy(outptr2, ptr2, size2 * elemsize2);

    return 0;

}

cv::Mat ncnn2cv(ncnn::Mat in, bool show= false){
    cv::Mat out(in.h, in.w, CV_8UC3);
    in.to_pixels(out.data, ncnn::Mat::PIXEL_BGR);
    if (show){
        cv::imshow("ncnn2cv", out);
        cv::waitKey(0);
    }
    return out;
}

ncnn::Mat cv2ncnn(cv::Mat img)
{
    ncnn::Mat out;
//    out = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR2RGB, img.cols, img.rows, 112, 112);
    out = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_BGR2RGB, img.cols, img.rows);
    return out;
}

void pretty_print(const ncnn::Mat& m)
{
    for (int q=0; q<m.c; q++)
    {
        const float* ptr = m.channel(q);
        for (int y=0; y<m.h; y++)
        {
            for (int x=0; x<m.w; x++)
            {
                printf("%f ", ptr[x]);
            }
            ptr += m.w;
            printf("\n");
        }
        printf("------------------------\n");
        break;
    }
    printf("$$$$$$$$$$$$$$$$$$$$$$\n");
}

void tiny_print(const ncnn::Mat& m)
{

    for(int i = 0; i < 10; i ++)
    {
        printf("%f ", m[i]);
    }
    printf("\n");
}

ncnn::Mat crop_resize(const ncnn::Mat &m, int xmin, int ymin, int xmax, int ymax)
{
//    m.data;
}

int main(int argc, char** argv)
{

    timer.tic();
    string detector_param = "../model/faceDetector.param";
    string detector_bin = "../model/faceDetector.bin";
    const int max_side = 320;
    // slim or RFB
    Detector detector(detector_param, detector_bin, false);
    timer.toc("init detect model");

    timer.tic();
    string alive_param = "../model/FaceAnti-Spoofing.param";
    string alive_bin = "../model/FaceAnti-Spoofing.bin";
    ncnn::Net alive;
    alive.load_param(alive_param.c_str());
    alive.load_model(alive_bin.c_str());
    timer.toc("init alive model");

    timer.tic();

    cv::Mat img1, img2, img, combine;
    ncnn::Mat imgn1, imgn2, imgn, combinen, out, in;
    bool flag1, flag2;

//    img1 = cv::imread("../data/b_0000001193_0.png");
//    img2 = cv::imread("../data/b_0000001193_1.png");

//    img1 = cv::imread("../data/a_0000000001_0.png");
//    img2 = cv::imread("../data/a_0000000001_1.png");

    img1 = cv::imread("../data/b_0000007288_0.png");
    img2 = cv::imread("../data/b_0000007288_1.png");

    img = img1;



    timer.toc("hello");

    cv::imshow("cap1", img1);
    cv::imshow("cap2", img2);

    // scale
    float long_side = std::max(img.cols, img.rows);
    float scale = max_side/long_side;
    cv::Mat img_scale;
    cv::Size size = cv::Size(img.cols*scale, img.rows*scale);
    cv::resize(img, img_scale, cv::Size(img.cols*scale, img.rows*scale));

    std::vector<bbox> boxes;

    timer.tic();

    detector.Detect(img_scale, boxes);
    timer.toc("----total timer:");

    const float mean_vals[4] = {123.675, 116.28, 103.53, 123.675};
    const float norm_vals[4] = {1/58.395f, 1/57.12f, 1/57.37f, 1/58.395f};

    imgn1 = cv2ncnn(img1);
    imgn2 = cv2ncnn(img2);

//    pretty_print(imgn1);
//    imgn1.fill(1.0f);
//    imgn2.fill(0.0f);
    cat(imgn1, imgn2, combinen);

//    combine

//    combinen.fill(1.f);
    combinen.substract_mean_normalize(mean_vals, norm_vals);
    pretty_print(combinen);
    tiny_print(combinen);




    timer.tic();
    ncnn::Extractor ex = alive.create_extractor();
    ex.input("input.1", combinen);
    ex.extract("466", out);
    timer.toc("alive infer");
    pretty_print(out);



    draw(img, boxes, scale);
    draw(img2, boxes, scale);

    cv::imshow("cap1_det", img1);
    cv::imshow("cap2_det", img2);

    cv::waitKey();

    return 0;
}
