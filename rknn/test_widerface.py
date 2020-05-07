from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_slim, cfg_rfb
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from models.net_slim import Slim
from models.net_rfb import RFB
from utils.box_utils import decode, decode_landm
from utils.timer import Timer
from rknn.api import RKNN


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('-m', '--trained_model', default='./weights_RFB/RBF_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='RFB', help='Backbone network mobile0.25 or slim or RFB')
parser.add_argument('--origin_size', default=False, type=str, help='Whether use origin image size to evaluate')
parser.add_argument('--long_side', type=int, default=320, help='when origin_size is false, long_side is scaled size(320 or 640 for long side)')
parser.add_argument('--save_folder', default='./widerface_evaluate/widerface_txt/', type=str, help='Dir to save txt results')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--dataset_folder', default='./data/widerface/val/images/', type=str, help='dataset path')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=False, help='show detection results')
parser.add_argument('--vis_thres', default=0.1, type=float, help='visualization_threshold')
parser.add_argument('--rknn_platform', default='rk1109')
parser.add_argument('--device_id', default='c3d9b8674f4b94f6')

# parser.add_argument('--target_platform', )
# parser.add_argument('--device_id', )
args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def init_rknn(args):
    rknn = RKNN()
    print('--> config model')
    rknn.config(channel_mean_value='104 117 123 1', reorder_channel='2 1 0', target_platform=args.rknn_platform)
    print('done')
    print('--> Loading model')
    ret = rknn.load_rknn(args.trained_model)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')
    print('--> Init runtime environment')
    ret = rknn.init_runtime(target=args.rknn_platform, device_id=args.device_id)
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')
    return rknn


if __name__ == '__main__':
    torch.set_grad_enabled(False)

    cfg = cfg_rfb
    # net = None
    # if args.network == "mobile0.25":
    #     cfg = cfg_mnet
    #     net = RetinaFace(cfg = cfg, phase = 'test')
    # elif args.network == "slim":
    #     cfg = cfg_slim
    #     net = Slim(cfg = cfg, phase = 'test')
    # elif args.network == "RFB":
    #     cfg = cfg_rfb
    #     net = RFB(cfg = cfg, phase = 'test')
    # else:
    #     print("Don't support network!")
    #     exit(0)

    # net = load_model(net, args.trained_model, args.cpu)
    # net.eval()

    rknn = init_rknn(args)

    print('Finished loading model!')
    # print(net)
    cudnn.benchmark = True
    # device = torch.device("cpu" if args.cpu else "cuda")
    # net = net.to(device)

    # testing dataset
    testset_folder = args.dataset_folder
    testset_list = args.dataset_folder[:-7] + "wider_val.txt"

    with open(testset_list, 'r') as fr:
        test_dataset = fr.read().split()
    num_images = len(test_dataset)

    _t = {'forward_pass': Timer(), 'misc': Timer()}

    # testing begin
    for i, img_name in enumerate(test_dataset):

        image_path = testset_folder + img_name
        # image_path = 'img/sample.jpg'
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)

        h_ = (480 if args.long_side == 640 else 240)
        w_ = args.long_side

        h_raw = img_raw.shape[0]
        w_raw = img_raw.shape[1]


        h_scale = img_raw.shape[0] / h_
        w_scale = img_raw.shape[1] / w_
        img_scale = cv2.resize(img_raw, (w_, h_))

        img = np.float32(img_scale)

        # testing scale
        target_size = args.long_side
        max_size = args.long_side
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        resize = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(resize * im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)
        if args.origin_size:
            resize = 1

        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        im_height, im_width, _ = img.shape
        # scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        scale = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        # img -= (104, 117, 123)
        # img = img.transpose(2, 0, 1)
        # img = torch.from_numpy(img).unsqueeze(0)
        # img = img.to(device)
        # scale = scale.to(device)

        _t['forward_pass'].tic()
        # loc, conf, landms = net(img)  # forward pass

        outputs = rknn.inference(inputs=[img])

        loc, conf, landms = outputs

        _t['forward_pass'].toc()
        _t['misc'].tic()
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        # priors = priors.numpy()
        # priors = priors.to(device)
        prior_data = priors.data.numpy()
        boxes = decode(loc.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        # boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0)[:, 1]
        landms = decode_landm(landms.squeeze(0), prior_data, cfg['variance'])
        # scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
        #                        img.shape[3], img.shape[2], img.shape[3], img.shape[2],
        #                        img.shape[3], img.shape[2]])
        # scale1 = scale1.to(device)
        scale1 = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0],
                           img.shape[1], img.shape[0], img.shape[1], img.shape[0],
                           img.shape[1], img.shape[0],])
        landms = landms * scale1 / resize
        # landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        # order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        # dets = dets[:args.keep_top_k, :]
        # landms = landms[:args.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)
        _t['misc'].toc()

        # --------------------------------------------------------------------
        save_name = args.save_folder + img_name[:-4] + ".txt"
        dirname = os.path.dirname(save_name)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        dets[:, 0:4:2] = dets[:, 0:4:2] / w_ * w_raw
        dets[:, 1:4:2] = dets[:, 1:4:2] / h_ * h_raw

        dets[:, 5:15:2] = dets[:, 5:15:2] / w_ * w_raw
        dets[:, 6:15:2] = dets[:, 6:15:2] / h_ * h_raw
        # box[0:4:2] = box[0:4:2] / w_ * w_raw
        # box[1:4:2] = box[1:4:2] / h_ * h_raw

        with open(save_name, "w") as fd:
            bboxs = dets
            file_name = os.path.basename(save_name)[:-4] + "\n"
            bboxs_num = str(len(bboxs)) + "\n"
            fd.write(file_name)
            fd.write(bboxs_num)
            for box in bboxs:

                x = int(box[0])
                y = int(box[1])
                w = int(box[2]) - int(box[0])
                h = int(box[3]) - int(box[1])
                confidence = str(box[4])
                line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n"
                fd.write(line)

        print('im_detect: {:d}/{:d} forward_pass_time: {:.4f}s misc: {:.4f}s'.format(i + 1, num_images, _t['forward_pass'].average_time, _t['misc'].average_time))

        # save image
        if args.save_image:
            for b in dets:
                if b[4] < args.vis_thres:
                    continue
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(img_raw, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                # landms
                cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
            # save image
            if not os.path.exists("./results/"):
                os.makedirs("./results/")
            name = "./results/" + str(i) + ".jpg"
            cv2.imwrite(name, img_raw)
        # break
