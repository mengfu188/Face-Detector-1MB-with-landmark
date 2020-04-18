//
// Created by cmf on 20-4-13.
//

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "net.h"

void pretty_print(const ncnn::Mat &m) {
    for (int q = 0; q < m.c; q++) {
        const float *ptr = m.channel(q);
        for (int y = 0; y < m.h; y++) {
            for (int x = 0; x < m.w; x++) {
                printf("%f ", ptr[x]);
            }
            ptr += m.w;
            printf("\n");
        }
        printf("------------------------\n");
    }
}

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
    cv::waitKey(0);
}

static ncnn::Mat IntArrayMat(int a0) {
    ncnn::Mat m(1);
    int *p = m;
    p[0] = a0;
    return m;
}

static ncnn::Mat IntArrayMat(int a0, int a1) {
    ncnn::Mat m(2);
    int *p = m;
    p[0] = a0;
    p[1] = a1;
    return m;
}

static ncnn::Mat IntArrayMat(int a0, int a1, int a2) {
    ncnn::Mat m(3);
    int *p = m;
    p[0] = a0;
    p[1] = a1;
    p[2] = a2;
    return m;
}

static void print_int_array(const ncnn::Mat &a) {
    const int *pa = a;

    fprintf(stderr, "[");
    for (int i = 0; i < a.w; i++) {
        fprintf(stderr, " %d", pa[i]);
    }
    fprintf(stderr, " ]");
}


ncnn::Layer *build_crop() {

    ncnn::ParamDict pd;

    ncnn::Mat _starts = IntArrayMat(0, 0, 0);
    ncnn::Mat _ends = IntArrayMat(3, 56, 112);
    pd.set(9, _starts);
    pd.set(10, _ends);

    int typeindex = ncnn::layer_to_index("Crop");
    ncnn::Layer *op = ncnn::create_layer(typeindex);
    op->load_param(pd);
    return op;
}

void test_crop(ncnn::Mat &in) {

    ncnn::Mat a = IntArrayMat(1, 2, 3);
//    pretty_print(a);
    int32_t *p = a;
    for (int i = 0; i < 4; i++) {
        printf("%d ", p[i]);
    }
    print_int_array(a);

    ncnn::Layer *op = build_crop();
    ncnn::Mat out;
    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = true;
    opt.use_int8_inference = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_int8_storage = false;
    opt.use_int8_arithmetic = false;
    op->forward(in, out, opt);
//    pretty_print(out);
    visualize("b", out);
}

int main() {
    cv::Mat img = cv::imread("../sample.jpg");
    int w = img.cols;
    int h = img.rows;
    printf("w is %d, h is %d\n", w, h);

    // subtract 128, norm to -1 ~ 1
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR, w, h, 112, 112);
    visualize("title", in);
    float mean[1] = {128.f};
    float norm[1] = {1 / 128.f};
    in.substract_mean_normalize(mean, norm);

    ncnn::Net net;
    net.load_param("../model/faceRecognize.param");
    net.load_model("../model/faceRecognize.bin");

    ncnn::Extractor ex = net.create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(4);
    in.fill(1.f);
    ex.input("data", in);

    ncnn::Mat feat;
    ex.extract("fc1", feat);

    pretty_print(feat);
    test_crop(in);
    return 0;
}