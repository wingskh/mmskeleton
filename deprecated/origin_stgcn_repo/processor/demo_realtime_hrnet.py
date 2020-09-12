#!/usr/bin/env python
import os
import sys
import argparse
import json
import shutil
import time
import ast
import threading

import numpy as np
import torch
import skvideo.io

from .io import IO
import tools
import tools.utils as utils
from tools.utils.live import Stack
from tools.utils.live import capture_thread
# TODO import HRNet_model
from pose_estimator.simple_HRNet.SimpleHRNet import SimpleHRNet
import cv2


class DemoRealtime(IO):
    """ A demo for utilizing st-gcn in the realtime action recognition.
    The Openpose python-api is required for this demo.
    Since the pre-trained model is trained on videos with 30fps,
    and Openpose is hard to achieve this high speed in the single GPU,
    if you want to predict actions by **camera** in realtime,
    either data interpolation or new pre-trained model
    is required.
    Pull requests are always welcome.
    """

    def start(self):
        # load openpose python api
        # if self.arg.openpose is not None:
        #     sys.path.append('{}/python'.format(self.arg.openpose))
        #     sys.path.append('{}/build/python'.format(self.arg.openpose))
        # try:
        #     from openpose import pyopenpose as op
        # except:
        #     print('Can not find Openpose Python API.')
        #     return

        label_name_path = './resource/hrnet/label_name.txt'
        with open(label_name_path) as f:
            label_name = f.readlines()
            label_name = [line.rstrip() for line in label_name]
            self.label_name = label_name

        # initiate
        # opWrapper = op.WrapperPython()
        # params = dict(model_folder='./models', model_pose='COCO')
        # opWrapper.configure(params)
        # opWrapper.start()
        # TODO pose_model is simple-HRNet
        image_resolution = ast.literal_eval(self.arg.image_resolution)
        pose_model = SimpleHRNet(
            self.arg.hrnet_c,
            self.arg.hrnet_j,
            self.arg.hrnet_weights,
            resolution=image_resolution,
            multiperson=not self.arg.single_person,
            max_batch_size=self.arg.max_batch_size,
            yolo_model_def=self.arg.yolo_model_def,
            yolo_weights_path=self.arg.yolo_weights_path,
            device=self.dev
        )
        self.model.eval()
        pose_tracker = naive_pose_tracker()

        print ("load video......")
        #if self.arg.video == 'camera_source':
        #    video_capture = cv2.VideoCapture(0)
        #else:
        #    video_capture = cv2.VideoCapture(self.arg.video)

        frame_buffer = Stack(3)
        lock = threading.RLock()
        t1 = threading.Thread(target=capture_thread, args=(self.arg.video, frame_buffer, lock))
        # t1 = threading.Thread(target=capture_thread, args=(self.arg.video, frame_buffer))
        t1.start()

        # start recognition
        start_time = time.time()
        frame_index = 0
        while (True):

            tic = time.time()
            
            if frame_buffer.size() > 0:
                lock.acquire()
                frame = frame_buffer.pop()
                lock.release()
                now = time.time()
                #print("start a frame: {}".format(now))
            else:
                # print("continue")
                continue
            
            # get image
            # video_capture = cv2.VideoCapture(self.arg.video)
            # ret, orig_image = video_capture.read()
            # print(orig_image)
            orig_image = frame
            if orig_image is None:
                break
            # source_H, source_W, _ = orig_image.shape
            # orig_image = cv2.resize(
            #     orig_image, (256 * source_W // source_H, 256))
            # H, W, _ = orig_image.shape
            H, W, _ = orig_image.shape

            # pose estimation
            # datum = op.Datum()
            # datum.cvInputData = orig_image
            # opWrapper.emplaceAndPop([datum])
            # multi_pose = datum.poseKeypoints  
            multi_pose_17 = pose_model.predict(orig_image)  # multi_pose: (num_person, num_joint, 3)
            now1 = time.time()
            #print("pose estimation a frame: {}".format(now1-tic))
            # TODO 17 joints -> 18 joints
            if multi_pose_17.shape[0] > 0:
                neck = 0.5 * (multi_pose_17[:, 5, :] + multi_pose_17[:, 6, :])  # neck is the mean of shoulders
                neck = np.expand_dims(neck, 1)  # shape: [n, 3] -> [n, 1, 3]
                # print('neck.shape: {}'.format(neck.shape))
                multi_pose_18 = np.concatenate((multi_pose_17, neck), 1)
                # print('multi_pose_18.shape: {}'.format(multi_pose_18.shape))
                # convert coco format to openpose format
                # convert coco format to openpose format
                multi_pose = multi_pose_18[:, [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3], :]
            else:
                continue
            if len(multi_pose.shape) != 3:
                continue

            # normalization
            # convert x and y
            multi_pose = multi_pose[:, :, [1, 0, 2]]
            multi_pose[:, :, 0] = multi_pose[:, :, 0] / W
            multi_pose[:, :, 1] = multi_pose[:, :, 1] / H
            multi_pose[:, :, 0:2] = multi_pose[:, :, 0:2] - 0.5
            multi_pose[:, :, 0][multi_pose[:, :, 2] == 0] = 0
            multi_pose[:, :, 1][multi_pose[:, :, 2] == 0] = 0
            now2 = time.time()
            print("convert a frame: {}".format(now2-now1))

            # pose tracking
            if self.arg.video == 'camera_source':
                frame_index = int((time.time() - start_time) * self.arg.fps)
            else:
                frame_index += 1
            pose_tracker.update(multi_pose, frame_index)
            data_numpy = pose_tracker.get_skeleton_sequence()
            print("get_skeleton_sequence: {} frame".format(data_numpy.shape[1]))
            data = torch.from_numpy(data_numpy)
            data = data.unsqueeze(0)
            data = data.float().to(self.dev).detach()  # (1, channel, frame, joint, person)
            now3 = time.time()
            print("pose track a frame: {}".format(now3-now2))

            # model predict
            voting_label_name, video_label_name, output, intensity = self.predict(
                data)
            now4 = time.time()
            print("action recognition a frame: {}".format(now4-now3))

            # visualization
            app_fps = 1 / (time.time() - tic)
            print("fps:{}".format(app_fps))
            
            image = self.render(data_numpy, voting_label_name,
                                video_label_name, intensity, orig_image, app_fps)
            now5 = time.time()
            print("render a frame: {}".format(now5-now4))

            #cv2.imshow("ST-GCN", image)

            now6 = time.time()
            print("visualize a frame: {}".format(now6-now5))
            
            print("action class:{}\n".format(voting_label_name))
            # save image
            #if voting_label_name == "clean_and_jerk":
            if voting_label_name:
                #image = self.render(data_numpy, voting_label_name, video_label_name, intensity, orig_image, app_fps)
                action_path = os.path.join("./action", voting_label_name+"_"+str(tic)+".png")
                cv2.imwrite(action_path, image)
                cv2.imwrite("action.png",image)
            
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
            # video_capture.release()
            # print("1111111111")

    def predict(self, data):
        # forward
        output, feature = self.model.extract_feature(data)
        output = output[0]
        feature = feature[0]
        intensity = (feature * feature).sum(dim=0) ** 0.5
        intensity = intensity.cpu().detach().numpy()

        # get result
        # classification result of the full sequence
        voting_label = output.sum(dim=3).sum(
            dim=2).sum(dim=1).argmax(dim=0)
        voting_label_name = self.label_name[voting_label]
        # classification result for each person of the latest frame
        num_person = data.size(4)
        latest_frame_label = [output[:, :, :, m].sum(
            dim=2)[:, -1].argmax(dim=0) for m in range(num_person)]
        latest_frame_label_name = [self.label_name[l]
                                   for l in latest_frame_label]

        num_person = output.size(3)
        num_frame = output.size(1)
        video_label_name = list()
        for t in range(num_frame):
            frame_label_name = list()
            for m in range(num_person):
                person_label = output[:, t, :, m].sum(dim=1).argmax(dim=0)
                person_label_name = self.label_name[person_label]
                frame_label_name.append(person_label_name)
            video_label_name.append(frame_label_name)
        return voting_label_name, video_label_name, output, intensity

    def render(self, data_numpy, voting_label_name, video_label_name, intensity, orig_image, fps=0):
        images = utils.visualization.stgcn_visualize(
            data_numpy[:, [-1]],
            self.model.graph.edge,
            intensity[[-1]], [orig_image],
            voting_label_name,
            [video_label_name[-1]],
            self.arg.height,
            fps=fps)
        image = next(images)
        image = image.astype(np.uint8)
        return image

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = IO.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Demo for Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        parser.add_argument('--video',
                            default='./resource/media/skateboarding.mp4',
                            help='Path to video')
        parser.add_argument('--openpose',
                            default=None,
                            help='Path to openpose')
        parser.add_argument('--model_input_frame',
                            default=128,
                            type=int)
        parser.add_argument('--model_fps',
                            default=30,
                            type=int)
        parser.add_argument('--height',
                            default=1080,
                            type=int,
                            help='height of frame in the output video.')
        parser.add_argument("--action",
                            help="the action name want to save",
                            type=str,
                            default="clean_and_jerk")

        # HRNet arguments
        parser.add_argument("--hrnet_c",
                            help="hrnet parameters - number of channels",
                            type=int,
                            default=32)
        parser.add_argument("--hrnet_j", "-j",
                            help="hrnet parameters - number of joints",
                            type=int,
                            default=17)
        parser.add_argument("--hrnet_weights",
                            help="hrnet parameters - path to the pretrained weights",
                            type=str,
                            default="./pose_estimator/simple_HRNet/weights/pose_hrnet_w32_256x192.pth")
        parser.add_argument("--image_resolution", "-r",
                            help="image resolution of HRNet input",
                            type=str,
                            default='(256, 192)')
        parser.add_argument("--single_person",
                            help="disable the multiperson detection (YOLOv3 or an equivalen detector is required for"
                                 "multiperson detection)",
                            action="store_true")
        parser.add_argument("--max_batch_size",
                            help="maximum batch size used for inference, used for multiperson detector YOLOv3",
                            type=int,
                            default=16)
        parser.add_argument("--yolo_model_def",
                            help="path to yolo model definition file",
                            type=str,
                            default="./pose_estimator/simple_HRNet/models/detectors/yolo/config/yolov3-tiny.cfg")
        parser.add_argument("--yolo_weights_path",
                            help="path to yolo pretrained weights file",
                            type=str,
                            default="./pose_estimator/simple_HRNet/models/detectors/yolo/weights/yolov3-tiny.weights")
        # argument "--device" is duplicate with IO (which is the parent class of this class, in io.py)
        # parser.add_argument("--device",
        #                     help="device to be used (default: cuda, if available)",
        #                     type=str,
        #                     default=None)

        # st-gcn default settings
        parser.set_defaults(
            config='./config/st_gcn/kinetics-skeleton/demo_realtime_hrnet.yaml')
        parser.set_defaults(print_log=False)
        # endregion yapf: enable

        return parser


class naive_pose_tracker():
    """ A simple tracker for recording person poses and generating skeleton sequences.
    For actual occasion, I recommend you to implement a robuster tracker.
    Pull-requests are welcomed.
    """

    def __init__(self, data_frame=128, 7, max_frame_dis=np.inf):
        self.data_frame = data_frame
        self.num_joint = num_joint
        self.max_frame_dis = max_frame_dis
        self.latest_frame = 0
        self.trace_info = list()

    def update(self, multi_pose, current_frame):
        # multi_pose.shape: (num_person, num_joint, 3)

        if current_frame <= self.latest_frame:
            return

        if len(multi_pose.shape) != 3:
            return

        score_order = (-multi_pose[:, :, 2].sum(axis=1)).argsort(axis=0)
        for p in multi_pose[score_order]:

            # match existing traces
            matching_trace = None
            matching_dis = None
            for trace_index, (trace, latest_frame) in enumerate(self.trace_info):
                # trace.shape: (num_frame, num_joint, 3)
                if current_frame <= latest_frame:
                    continue
                mean_dis, is_close = self.get_dis(trace, p)
                if is_close:
                    if matching_trace is None:
                        matching_trace = trace_index
                        matching_dis = mean_dis
                    elif matching_dis > mean_dis:
                        matching_trace = trace_index
                        matching_dis = mean_dis

            # update trace information
            if matching_trace is not None:
                trace, latest_frame = self.trace_info[matching_trace]

                # padding zero if the trace is fractured
                pad_mode = 'interp' if latest_frame == self.latest_frame else 'zero'
                pad = current_frame - latest_frame - 1
                new_trace = self.cat_pose(trace, p, pad, pad_mode)
                self.trace_info[matching_trace] = (new_trace, current_frame)

            else:
                new_trace = np.array([p])
                self.trace_info.append((new_trace, current_frame))

        self.latest_frame = current_frame

    # TODO how many frames are included
    def get_skeleton_sequence(self):

        # remove old traces
        valid_trace_index = []
        for trace_index, (trace, latest_frame) in enumerate(self.trace_info):
            if self.latest_frame - latest_frame < self.data_frame:
                valid_trace_index.append(trace_index)
        self.trace_info = [self.trace_info[v] for v in valid_trace_index]

        num_trace = len(self.trace_info)
        if num_trace == 0:
            return None

        data = np.zeros((3, self.data_frame, self.num_joint, num_trace))
        for trace_index, (trace, latest_frame) in enumerate(self.trace_info):
            end = self.data_frame - (self.latest_frame - latest_frame)
            d = trace[-end:]
            beg = end - len(d)
            data[:, beg:end, :, trace_index] = d.transpose((2, 0, 1))

        return data

    # concatenate pose to a trace
    def cat_pose(self, trace, pose, pad, pad_mode):
        # trace.shape: (num_frame, num_joint, 3)
        num_joint = pose.shape[0]
        num_channel = pose.shape[1]
        if pad != 0:
            if pad_mode == 'zero':
                trace = np.concatenate(
                    (trace, np.zeros((pad, num_joint, 3))), 0)
            elif pad_mode == 'interp':
                last_pose = trace[-1]
                coeff = [(p + 1) / (pad + 1) for p in range(pad)]
                interp_pose = [(1 - c) * last_pose + c * pose for c in coeff]
                trace = np.concatenate((trace, interp_pose), 0)
        new_trace = np.concatenate((trace, [pose]), 0)
        return new_trace

    # calculate the distance between a existing trace and the input pose

    def get_dis(self, trace, pose):
        last_pose_xy = trace[-1, :, 0:2]
        curr_pose_xy = pose[:, 0:2]

        mean_dis = ((((last_pose_xy - curr_pose_xy) ** 2).sum(1)) ** 0.5).mean()
        wh = last_pose_xy.max(0) - last_pose_xy.min(0)
        scale = (wh[0] * wh[1]) ** 0.5 + 0.0001
        is_close = mean_dis < scale * self.max_frame_dis
        return mean_dis, is_close