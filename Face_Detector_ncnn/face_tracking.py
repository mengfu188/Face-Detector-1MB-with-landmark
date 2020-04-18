import cv2
import numpy as np
import matplotlib.pyplot as plt
# import demo12.kcf_demo as kcf
import time
from mtcnn.mtcnn import MTCNN


cv2.useOptimized()
cv2.setUseOptimized(True)
cv2.setNumThreads(4)


class Tracker(object):
    '''
    追踪者模块,用于追踪指定目标
    '''
    def __init__(self,tracker_type = "BOOSTING",draw_coord = True):
        '''
        初始化追踪器种类
        '''
        #获得opencv版本
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
        print((major_ver, minor_ver, subminor_ver))
        self.tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
        self.tracker_type = tracker_type
        self.isWorking = False
        self.draw_coord = draw_coord
        #构造追踪器
        if int(major_ver) < 3:
            pass
            # self.tracker = cv2.Tracker_create(tracker_type)
        else:
            if tracker_type == 'BOOSTING':
                self.tracker = cv2.TrackerBoosting_create()
            if tracker_type == 'MIL':
                self.tracker = cv2.TrackerMIL_create()
            if tracker_type == 'KCF':
                self.tracker = cv2.TrackerKCF_create()
            if tracker_type == 'TLD':
                self.tracker = cv2.TrackerTLD_create()
            if tracker_type == 'MEDIANFLOW':
                self.tracker = cv2.TrackerMedianFlow_create()
            if tracker_type == 'GOTURN':
                self.tracker = cv2.TrackerGOTURN_create()
    def initWorking(self,frame,box):
        '''
        追踪器工作初始化
        frame:初始化追踪画面
        box:追踪的区域
        '''
        if not self.tracker:
            raise Exception("追踪器未初始化")
        status = self.tracker.init(frame,box)
        if not status:
            raise Exception("追踪器工作初始化失败")
        self.coord = box
        self.isWorking = True

    def track(self, frame):
        '''
        开启追踪
        '''
        # message = None
        tracking_flag = 1
        if self.isWorking:
            status, self.coord = self.tracker.update(frame)
            if status:
                # message = {"coord": [((int(self.coord[0]), int(self.coord[1])),
                #                       (int(self.coord[0] + self.coord[2]), int(self.coord[1] + self.coord[3])))]}
                tracking_flag = 1
                if self.draw_coord:
                    p1 = (int(self.coord[0]), int(self.coord[1]))
                    p2 = (int(self.coord[0] + self.coord[2]), int(self.coord[1] + self.coord[3]))
                    # print(self.coord)
                    cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
                    # message['msg'] = "is tracking"
                    cv2.putText(frame, 'is tracking', (64, 64+60), 1, 1, (255, 0, 0))

            else:
                tracking_flag = 0
                cv2.putText(frame, 'Lost', (64, 64 + 60), 1, 1, (255, 0, 0))
                print('==========================>Lost')
        # return MessageItem(frame, message)
        return frame, tracking_flag


def detect_face(capture):
    max_face_area = 0
    max_indx = 0
    flag = 0
    # rect = []

    while flag == 0:
        ret, frame = capture.read()
        if ret:
            faces = detector.detect_faces(frame)
            if len(faces) >= 1:
                flag = 1
                for i in range(len(faces)):
                    if faces[i]['box'][2] * faces[i]['box'][3] > max_face_area:
                        max_face_area = faces[i]['box'][2] * faces[i]['box'][3]
                        max_indx = i
                rect = (faces[max_indx]['box'][0], faces[max_indx]['box'][1], faces[max_indx]['box'][2], faces[max_indx]['box'][3])
                # for i in range(4):
                #     rect.append(faces[max_indx]['box'][i])
        else:
            print("can't open the camera!")

    return frame, rect



if __name__ == '__main__':
    detect_faces_flag = 0


    # cap = cv2.VideoCapture(0)
    # assert capture.open('/home/cmf/t.avi')
    cap = cv2.VideoCapture('/home/cmf/t.avi')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # frame = np.zeros(shape=[int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    #                         int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
    #                         3], dtype=np.uint8)
    #

    detector = MTCNN()                                              # MTCNN初始化
    # a = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
    tracker = Tracker(tracker_type="MEDIANFLOW")                         # 定义跟踪算法
    start_time = time.time()
    frame, rect = detect_face(cap)                    # 一开始先捕获一张人脸
    print('捕获人脸耗时：', (time.time() - start_time))


    # rect = cv2.selectROI('Choose object', frame, False, False)
    start_time = time.time()
    tracker.initWorking(frame, rect)                                     # 跟踪器初始化
    print('初始化耗时：', (time.time()-start_time))
    # r = [rect[0] + rect[2], rect[1] + rect[3]]

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("can't open the camera!")
            break

        t_start = time.time()
        frame, tracking_flag = tracker.track(frame)
        t_stop = time.time()
        fps = int(1.0/(t_stop - t_start))

        # 人脸丢失，重新捕获
        if tracking_flag == 0:
            frame, rect = detect_face(cap)         # 重新捕获一张人脸
            tracker = Tracker(tracker_type="MEDIANFLOW")  # 定义跟踪算法
            tracker.initWorking(frame, rect)        # 重新初始化跟踪器
        # cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 255), 2)
        cv2.putText(frame, '#' + str(count + 1), (64, 64), 1, 1, (255, 0, 0))
        cv2.putText(frame, '{}fps'.format(fps), (64, 64+30), 1, 1, (255, 0, 0))
        count += 1
        cv2.imshow('kcf', frame)
        if cv2.waitKey(1) == 'q':
            break

