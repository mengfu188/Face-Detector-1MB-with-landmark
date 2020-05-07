import numpy as np
import cv2
from rknn.api import RKNN
import torchvision.models as models
import torch

import argparse
import time

import cv2
import torch
from deploy.ssd_detector import Detector

input_size = (240, 320)
# input_size = (480, 640)
NEED_BUILD_MODEL = True
QUAN = True
epochs = 1000
batch_size = 1
name = 'ssd_{}x{}_{}_{}_{}'.format(input_size[0], input_size[1], 8 if QUAN else 16, epochs, batch_size)
pt_model = 'deploy/ssd_{}x{}.pt'.format(input_size[0], input_size[1])
rknn_model = 'deploy/{}.rknn'.format(name)
print(pt_model)
input_size_list = [[3, input_size[0], input_size[1]]]
input_tensor = torch.Tensor(1, 3, input_size[0], input_size[1])
target = 'rk1109'
dataset = 'deploy/dataset.txt'


def get_args():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('-m', '--trained_model', default='weights/RFB_320_mask.pth',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--network', default='RFB', help='Backbone network mobile0.25 or slim or RFB')
    parser.add_argument('--origin_size', default=False, type=str, help='Whether use origin image size to evaluate')
    parser.add_argument('--long_side', default=320,
                        help='when origin_size is false, long_side is scaled size(320 or 640 for long side)')
    parser.add_argument('--cpu', action="store_true", default=True, help='Use cpu inference')
    parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
    parser.add_argument('--save_image', action="store_true", default=True, help='show detection results')
    parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')

    args = parser.parse_args()
    return args


def export_pytorch_model(args):
    detector = Detector(args)
    net = detector.net

    # net = models.resnet18(pretrained=True)
    net.eval()
    trace_model = torch.jit.trace(net, input_tensor)
    trace_model.save(pt_model)


def show_outputs(output):
    output_sorted = sorted(output, reverse=True)
    top5_str = '\n-----TOP 5-----\n'
    for i in range(5):
        value = output_sorted[i]
        index = np.where(output == value)
        for j in range(len(index)):
            if (i + j) >= 5:
                break
            if value > 0:
                topi = '{}: {}\n'.format(index[j], value)
            else:
                topi = '-1: 0.0\n'
            top5_str += topi
    print(top5_str)


def show_perfs(perfs):
    perfs = 'perfs: {}\n'.format(perfs)
    print(perfs)


if __name__ == '__main__':

    args = get_args()

    export_pytorch_model(args)

    # model = pt_model
    # input_size_list = [[3,224,224]]
    #
    # Create RKNN object
    rknn = RKNN()

    # pre-process config
    print('--> config model')
    rknn.config(channel_mean_value='104 117 123 1', reorder_channel='2 1 0', target_platform=target,
                batch_size=batch_size, epochs=epochs
                )
    print('done')

    # Load pytorch model
    print('--> Loading model')

    if NEED_BUILD_MODEL:
        ret = rknn.load_pytorch(model=pt_model, input_size_list=input_size_list)
        if ret != 0:
            print('Load pytorch model failed!')
            exit(ret)
        print('done')
        # Build model
        print('--> Building model')
        ret = rknn.build(do_quantization=QUAN, dataset=dataset)
        if ret != 0:
            print('Build pytorch failed!')
            exit(ret)
        print('done')

        # Export rknn model
        print('--> Export RKNN model')
        ret = rknn.export_rknn(rknn_model)
        if ret != 0:
            print('Export resnet_18.rknn failed!')
            exit(ret)
        print('done')

    ret = rknn.load_rknn(rknn_model)

    # Set inputs
    # img = cv2.imread('deploy/0_Parade_marchingband_1_5.jpg')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, (input_size, input_size))
    img = np.ones((3, input_size[0], input_size[1]), dtype=np.float32)

    # init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime(target='rk1109', device_id='c3d9b8674f4b94f6', perf_debug=True, eval_mem=True)
    # ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])

    # show_outputs(softmax(np.array(outputs[0][0])))
    print('done')

    # perf
    print('--> Begin evaluate model performance')
    perf_results = rknn.eval_perf(inputs=[img])
    print('done')

    rknn.eval_memory()

    rknn.release()
