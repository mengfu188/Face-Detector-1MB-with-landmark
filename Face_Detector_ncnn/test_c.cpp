//
// Created by cmf on 20-4-2.
//

#include <stdio.h>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <net.h>

using namespace std;

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
    out = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_BGR, img.cols, img.rows);
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

int main()
{
    cv::Mat img1, img2;
    ncnn::Mat ing1, ing2, in;
    int y, im_w, x, roiw, roih;

    img1 = cv::imread("../data/a_0000000001_0.png");
    cv::imshow("row img", img1);

    ing1 = cv2ncnn(img1);
    img2 = ncnn2cv(ing1);
    cv::imshow("rec row img", img2);
    cv::waitKey();
//    const unsigned char* data = ing1.data + (y * im_w + x) * 3;
//    in = ncnn::Mat::from_pixels(data, ncnn::PIXEL_RGB, roiw, roih, im_w * 3);
    ncnn::copy_cut_border(ing1, ing2, 100, 200, 100, 200);
    printf("width %d, height %d, channel %d", ing2.w, ing2.h, ing2.c);
//    img2 = ncnn2cv(ing2);
//    cv::imshow("crop img", img2);
//    cv::waitKey();

}