//
// Created by cmf on 20-4-13.
//

#include <string>

#include "net.h"


#include <vector>

#define TAG "LightFaceSo"

#include "FaceDetector.h"
#include <chrono>
#include <math.h>
#include <stack>
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <string>
#include <stack>
#include <chrono>
using namespace std::chrono;


//#include "FaceDetector.h"
static Timer timer;
using namespace std;

#if __ANDROID_API__ >= 9
#include <android/asset_manager.h>
#endif

class Recognize{
public:

#if __ANDROID_API__ >= 9

#endif
    Recognize(const string &param, const string &bin){
        this->net.load_param(param.c_str());
        this->net.load_model(bin.c_str());
    };

    float* recognize(ncnn::Mat in, float* landmark){
        timer.tic();
        timer.tic();
        in = preprocess(in, landmark);
        timer.toc("recognize preprocess");
#ifdef __linux__
//        visualize("face", in);
#endif
//        recognize_timer.toc("recognize preprocess cost");
//        LOGD("preprocess image width is %d, height is %d", in.w, in.h);

//        recognize_timer.tic();
        timer.tic();
        ncnn::Extractor ex = net.create_extractor();
        ex.set_light_mode(true);
        ex.set_num_threads(1);
//        in.fill(1.f);
        ex.input("data", in);
        ncnn::Mat out;
        ex.extract("fc1", out);
        timer.toc("face model infer");
//        recognize_timer.toc("face recognize infer cost");

        vector<float> feature;
        feature.resize(feature_dim);
        for (int i = 0; i < feature_dim; i++)
            feature[i] = out[i];
        float* embedding = normalize(feature);
        timer.toc("face total recognize ");
        return embedding;
    };

    ncnn::Mat preprocess(ncnn::Mat img, float* landmark) {
//    TODO Fix landmark
        int image_w = 112; //96 or 112
        int image_h = 112;

        float dst[10] = {30.2946, 65.5318, 48.0252, 33.5493, 62.7299,
                         51.6963, 51.5014, 71.7366, 92.3655, 92.2041};

        if (image_w == 112)
            for (int i = 0; i < 5; i++)
                dst[i] += 8.0;

        float src[10];
        for (int i = 0; i < 5; i++) {
            src[i] = landmark[2 * i];
            src[i + 5] = landmark[2 * i + 1];
        }

        float M[6];
        getAffineMatrix(src, dst, M);
        ncnn::Mat out;
        warpAffineMatrix(img, out, M, image_w, image_h);
        return out;
    }

private:
    ncnn::Net net;
    unsigned long feature_dim = 128;

#ifdef __linux__
    void visualize(const char *title, const ncnn::Mat &m) {
        std::vector<cv::Mat> normed_feats(m.c);

        for (int i = 0; i < m.c; i++) {
            cv::Mat tmp(m.h, m.w, CV_32FC1, (void *) (const float *) m.channel(i));

            cv::normalize(tmp, normed_feats[i], 0, 255, cv::NORM_MINMAX, CV_8U);

            cv::cvtColor(normed_feats[i], normed_feats[i], cv::COLOR_GRAY2BGR);

            // check NaN
            for (int y = 0; y < m.h; y++) {
                const float *tp = tmp.ptr<float>(y);
                uchar *sp = normed_feats[i].ptr<uchar>(y);
                for (int x = 0; x < m.w; x++) {
                    float v = tp[x];
                    if (v != v) {
                        sp[0] = 0;
                        sp[1] = 0;
                        sp[2] = 255;
                    }

                    sp += 3;
                }
            }
        }

        int tw = m.w < 10 ? 32 : m.w < 20 ? 16 : m.w < 40 ? 8 : m.w < 80 ? 4 : m.w < 160 ? 2 : 1;
        int th = (m.c - 1) / tw + 1;

        cv::Mat show_map(m.h * th, m.w * tw, CV_8UC3);
        show_map = cv::Scalar(127);

        // tile
        for (int i = 0; i < m.c; i++) {
            int ty = i / tw;
            int tx = i % tw;

            normed_feats[i].copyTo(show_map(cv::Rect(tx * m.w, ty * m.h, m.w, m.h)));
        }

        cv::resize(show_map, show_map, cv::Size(0, 0), 2, 2, cv::INTER_NEAREST);
        cv::imshow(title, show_map);
        cv::waitKey(1);
    }

#endif

    void getAffineMatrix(float* src_5pts, const float* dst_5pts, float* M)
    {
        float src[10], dst[10];
        memcpy(src, src_5pts, sizeof(float)*10);
        memcpy(dst, dst_5pts, sizeof(float)*10);

        float ptmp[2];
        ptmp[0] = ptmp[1] = 0;
        for (int i = 0; i < 5; ++i) {
            ptmp[0] += src[i];
            ptmp[1] += src[5+i];
        }
        ptmp[0] /= 5;
        ptmp[1] /= 5;
        for (int i = 0; i < 5; ++i) {
            src[i] -= ptmp[0];
            src[5+i] -= ptmp[1];
            dst[i] -= ptmp[0];
            dst[5+i] -= ptmp[1];
        }

        float dst_x = (dst[3]+dst[4]-dst[0]-dst[1])/2, dst_y = (dst[8]+dst[9]-dst[5]-dst[6])/2;
        float src_x = (src[3]+src[4]-src[0]-src[1])/2, src_y = (src[8]+src[9]-src[5]-src[6])/2;
        float theta = atan2(dst_x, dst_y) - atan2(src_x, src_y);

        float scale = sqrt(pow(dst_x, 2) + pow(dst_y, 2)) / sqrt(pow(src_x, 2) + pow(src_y, 2));
        float pts1[10];
        float pts0[2];
        float _a = sin(theta), _b = cos(theta);
        pts0[0] = pts0[1] = 0;
        for (int i = 0; i < 5; ++i) {
            pts1[i] = scale*(src[i]*_b + src[i+5]*_a);
            pts1[i+5] = scale*(-src[i]*_a + src[i+5]*_b);
            pts0[0] += (dst[i] - pts1[i]);
            pts0[1] += (dst[i+5] - pts1[i+5]);
        }
        pts0[0] /= 5;
        pts0[1] /= 5;

        float sqloss = 0;
        for (int i = 0; i < 5; ++i) {
            sqloss += ((pts0[0]+pts1[i]-dst[i])*(pts0[0]+pts1[i]-dst[i])
                       + (pts0[1]+pts1[i+5]-dst[i+5])*(pts0[1]+pts1[i+5]-dst[i+5]));
        }

        float square_sum = 0;
        for (int i = 0; i < 10; ++i) {
            square_sum += src[i]*src[i];
        }
        for (int t = 0; t < 200; ++t) {
            _a = 0;
            _b = 0;
            for (int i = 0; i < 5; ++i) {
                _a += ((pts0[0]-dst[i])*src[i+5] - (pts0[1]-dst[i+5])*src[i]);
                _b += ((pts0[0]-dst[i])*src[i] + (pts0[1]-dst[i+5])*src[i+5]);
            }
            if (_b < 0) {
                _b = -_b;
                _a = -_a;
            }
            float _s = sqrt(_a*_a + _b*_b);
            _b /= _s;
            _a /= _s;

            for (int i = 0; i < 5; ++i) {
                pts1[i] = scale*(src[i]*_b + src[i+5]*_a);
                pts1[i+5] = scale*(-src[i]*_a + src[i+5]*_b);
            }

            float _scale = 0;
            for (int i = 0; i < 5; ++i) {
                _scale += ((dst[i]-pts0[0])*pts1[i] + (dst[i+5]-pts0[1])*pts1[i+5]);
            }
            _scale /= (square_sum*scale);
            for (int i = 0; i < 10; ++i) {
                pts1[i] *= (_scale / scale);
            }
            scale = _scale;

            pts0[0] = pts0[1] = 0;
            for (int i = 0; i < 5; ++i) {
                pts0[0] += (dst[i] - pts1[i]);
                pts0[1] += (dst[i+5] - pts1[i+5]);
            }
            pts0[0] /= 5;
            pts0[1] /= 5;

            float _sqloss = 0;
            for (int i = 0; i < 5; ++i) {
                _sqloss += ((pts0[0]+pts1[i]-dst[i])*(pts0[0]+pts1[i]-dst[i])
                            + (pts0[1]+pts1[i+5]-dst[i+5])*(pts0[1]+pts1[i+5]-dst[i+5]));
            }
            if (abs(_sqloss - sqloss) < 1e-2) {
                break;
            }
            sqloss = _sqloss;
        }

        for (int i = 0; i < 5; ++i) {
            pts1[i] += (pts0[0] + ptmp[0]);
            pts1[i+5] += (pts0[1] + ptmp[1]);
        }

        M[0] = _b*scale;
        M[1] = _a*scale;
        M[3] = -_a*scale;
        M[4] = _b*scale;
        M[2] = pts0[0] + ptmp[0] - scale*(ptmp[0]*_b + ptmp[1]*_a);
        M[5] = pts0[1] + ptmp[1] - scale*(-ptmp[0]*_a + ptmp[1]*_b);
    }

    void warpAffineMatrix(ncnn::Mat src, ncnn::Mat &dst, float *M, int dst_w, int dst_h)
    {
        int src_w = src.w;
        int src_h = src.h;

        unsigned char * src_u = new unsigned char[src_w * src_h * 3]{0};
        unsigned char * dst_u = new unsigned char[dst_w * dst_h * 3]{0};

        src.to_pixels(src_u, ncnn::Mat::PIXEL_RGB);

        float m[6];
        for (int i = 0; i < 6; i++)
            m[i] = M[i];
        float D = m[0] * m[4] - m[1] * m[3];
        D = D != 0 ? 1./D : 0;
        float A11 = m[4] * D, A22 = m[0] * D;
        m[0] = A11; m[1] *= -D;
        m[3] *= -D; m[4] = A22;
        float b1 = -m[0] * m[2] - m[1] * m[5];
        float b2 = -m[3] * m[2] - m[4] * m[5];
        m[2] = b1; m[5] = b2;

        for (int y= 0; y < dst_h; y++)
        {
            for (int x = 0; x < dst_w; x++)
            {
                float fx = m[0] * x + m[1] * y + m[2];
                float fy = m[3] * x + m[4] * y + m[5];

                int sy = (int)floor(fy);
                fy -= sy;
                if (sy < 0 || sy >= src_h) continue;

                short cbufy[2];
                cbufy[0] = (short)((1.f - fy) * 2048);
                cbufy[1] = 2048 - cbufy[0];

                int sx = (int)floor(fx);
                fx -= sx;
                if (sx < 0 || sx >= src_w) continue;

                short cbufx[2];
                cbufx[0] = (short)((1.f - fx) * 2048);
                cbufx[1] = 2048 - cbufx[0];

                if (sy == src_h - 1 || sx == src_w - 1)
                    continue;
                for (int c = 0; c < 3; c++)
                {
                    dst_u[3 * (y * dst_w + x) + c] =
                            (
                                    src_u[3 * (sy * src_w + sx) + c] * cbufx[0] * cbufy[0] +
                                    src_u[3 * ((sy + 1) * src_w + sx) + c] * cbufx[0] * cbufy[1] +
                                    src_u[3 * (sy * src_w + sx + 1) + c] * cbufx[1] * cbufy[0] +
                                    src_u[3 * ((sy + 1) * src_w + sx + 1) + c] * cbufx[1] * cbufy[1]
                            ) >> 22;
                }
            }
        }

        dst = ncnn::Mat::from_pixels(dst_u, ncnn::Mat::PIXEL_BGR, dst_w, dst_h);
        delete[] src_u;
        delete[] dst_u;
    }

    float* normalize(vector<float> &feature) {
        float sum = 0;
        for (auto it = feature.begin(); it != feature.end(); it++)
            sum += (float) *it * (float) *it;
        sum = sqrt(sum);
        for (auto it = feature.begin(); it != feature.end(); it++)
            *it /= sum;
        float *ret = new float[feature.size()];
        for(int i = 0; i < feature.size(); i++)
            ret[i] = feature[i];
        return ret;
    }

};

int main(int argc, char **argv) {

    string det_param = "../model/faceDetector.param";
    string det_bin = "../model/faceDetector.bin";
    string rec_param = "../model/faceRecognize.param";
    string rec_bin = "../model/faceRecognize.bin";
//    rec_param = "../model/model-slim.param";
//    rec_bin = "../model/model-slim.bin";
//    rec_param = "../model/half/models-slim.param";
//    rec_bin = "../model/half/models-slim.bin";

    // slim or RFB
    Detector detector(det_param, det_bin, false);
    Recognize recognize(rec_param, rec_bin);

    // retinaface
    // Detector detector(param, bin, true);
    Timer timer;
    cv::VideoCapture cap("/home/cmf/t.avi");
    cv::Mat img;
    ncnn::Mat in;
    for (;;) {

        cap >> img;
        in = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_BGR, img.cols, img.rows);

        std::vector<bbox> boxes;

        timer.tic();

        detector.Detect(img, boxes);
        timer.toc("----total timer:");

        // draw image
        for (int j = 0; j < boxes.size(); ++j) {

            // recognize
            float* landmark = boxes[j].landmark();
            float * embedding = recognize.recognize(in, landmark);
            for(int k =0; k < 10; k++){
                printf("%f ", embedding[k]);
            }
            printf("\n");


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

