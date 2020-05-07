from __future__ import print_function

import argparse
import time

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from data import cfg_mnet, cfg_slim, cfg_rfb

from layers.functions.prior_box import PriorBox
from models.net_rfb import RFB
from models.net_slim import Slim
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.nms.py_cpu_nms import py_cpu_nms


def get_args():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('-m', '--trained_model', default='weights/RFB_320_mask.pth',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--network', default='RFB', help='Backbone network mobile0.25 or slim or RFB')
    parser.add_argument('--origin_size', default=False, type=str, help='Whether use origin image size to evaluate')
    parser.add_argument('--long_side', default=640,
                        help='when origin_size is false, long_side is scaled size(320 or 640 for long side)')
    parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
    parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=1000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
    parser.add_argument('--save_image', action="store_true", default=True, help='show detection results')
    parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
    args = parser.parse_args()
    return args


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


class Detector:
    def __init__(self, args):
        torch.set_grad_enabled(False)

        cfg = None
        net = None
        if args.network == "mobile0.25":
            cfg = cfg_mnet
            net = RetinaFace(cfg=cfg, phase='test')
        elif args.network == "slim":
            cfg = cfg_slim
            net = Slim(cfg=cfg, phase='test')
        elif args.network == "RFB":
            cfg = cfg_rfb
            net = RFB(cfg=cfg, phase='test')
        else:
            print("Don't support network!")
            exit(0)

        net = load_model(net, args.trained_model, args.cpu)
        net.eval()
        print('Finished loading model!')
        # print(net)
        cudnn.benchmark = True
        self.device = torch.device("cpu" if args.cpu else "cuda")
        self.net = net.to(self.device)
        self.cfg = cfg
        self.args = args

    def detect(self, img_raw, show=False):
        """

        :param img: bgr format
        :return:
        """
        img = self.preprocess(img_raw)
        tic = time.time()
        loc, conf, landms = self.net(img)  # forward pass
        # print('net forward time: {:.4f}'.format(time.time() - tic))
        dets = self.postprocess(img, loc, conf, landms)
        # dets shape is (N, 15)
        # [[xmin, ymin, xmax, ymax, score, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5]]
        if show:
            self.show(img_raw, dets)
        loc = dets[:, 0:4]
        conf = dets[:, 4]
        landms = dets[:, 5:]
        # landms.reshape(-1, 5, 2) [[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x5,y5]]
        return loc, conf, landms

    def preprocess(self, img_raw):
        """

        :param img_raw: bgr format
        :return:
        """
        img = np.float32(img_raw)

        # testing scale
        target_size = self.args.long_side
        max_size = self.args.long_side
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        resize = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(resize * im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)
        if self.args.origin_size:
            resize = 1

        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        im_height, im_width, _ = img.shape

        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        self.im_height = im_height
        self.im_width = im_width
        self.resize = resize
        return img

    def postprocess(self, img, loc, conf, landms):
        tic = time.time()
        scale = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2]])
        scale = scale.to(self.device)
        priorbox = PriorBox(self.cfg, image_size=(self.im_height, self.im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale / self.resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / self.resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:self.args.keep_top_k, :]
        landms = landms[:self.args.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)
        # print(f'net forward and post process time cost {time.time() - tic} s')

        return dets

    def show(self, img_raw, dets):
        for b in dets:
            if b[4] < self.args.vis_thres:
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

        name = "test.jpg"
        # cv2.imwrite(name, img_raw)
        cv2.imshow('test', img_raw)
        cv2.waitKey()


if __name__ == '__main__':
    args = get_args()
    face_detector = Detector(args)
    # while True:
    image_path = "./img/sample.jpg"

    img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
    dets = face_detector.detect(img_raw, True)
