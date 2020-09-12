#!/usr/bin/env python
import os
from os import walk
import sys
import argparse
import json
import shutil
import time
import math
import numpy as np
import torch
import skvideo.io
import sys
from deprecated.origin_stgcn_repo.processor.io import IO
import deprecated.origin_stgcn_repo.tools
import deprecated.origin_stgcn_repo.tools.utils as utils
import mmcv
from mmskeleton.apis import init_pose_estimator, inference_pose_estimator

import cv2


class DemoOffline(IO):

    def start(self):
        # initiate
        label_name_path = './data/label_name.txt'
        with open(label_name_path) as f:
            label_name = f.readlines()
            label_name = [line.rstrip() for line in label_name]
            self.label_name = label_name

        visualization = True
        # pose estimation

        if not visualization:
            files = []
            path = self.arg.video
            for (dirpath, dirnames, filenames) in walk(path):
                files.extend(filenames)
            files.sort()

            filename = 'label_output.csv'
            with open(filename, 'w') as fd:
                pass

            for file in files:
                video, data_numpy = self.pose_estimation(
                    visualization, path, file)

                # action recognition
                data = torch.from_numpy(data_numpy)
                data = data.unsqueeze(0)
                data = data.float().to(self.dev).detach()  # (1, channel, frame, joint, person)

                # model predict
                voting_label_name, video_label_name, output, intensity = self.predict(
                    data)

                with open(filename, 'a+') as fd:
                    fd.write(file+', '+voting_label_name+'\n')
        else:
            # for visualization
            # pose estimation
            video, data_numpy = self.pose_estimation(visualization)

            # action recognition
            data = torch.from_numpy(data_numpy)
            data = data.unsqueeze(0)
            data = data.float().to(self.dev).detach()  # (1, channel, frame, joint, person)

            # model predict
            voting_label_name, video_label_name, output, intensity = self.predict(
                data)

            # render the video
            images = self.render_video(data_numpy, voting_label_name,
                                       video_label_name, intensity, video)

            # visualize
            # fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            out = cv2.VideoWriter('visualized_video.avi',
                                  fourcc, 30.0, (1434, 1080))
            counter = 0
            for image in images:
                counter += 1
                image = image.astype(np.uint8)
                out.write(image)
            print("The total frame is:", counter, file=sys.stdout)
            out.release()

    def predict(self, data):
        # forward
        output, feature = self.model.extract_feature(data)
        output = output[0]
        feature = feature[0]
        intensity = (feature*feature).sum(dim=0)**0.5
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

    def render_video(self, data_numpy, voting_label_name, video_label_name, intensity, video):
        images = utils.visualization.stgcn_visualize(
            data_numpy,
            self.model.graph.edge,
            intensity, video,
            voting_label_name,
            video_label_name,
            self.arg.height)
        return images

    def pose_estimation(self, visualization, path="", filename=""):
        # load openpose python api
        if self.arg.openpose is not None:
            sys.path.append('{}/python'.format(self.arg.openpose))
            sys.path.append('{}/build/python'.format(self.arg.openpose))
        try:
            from openpose import pyopenpose as op
        except:
            print('Can not find Openpose Python API.')
            return

        # video_name = self.arg.video.split('/')[-1].split('.')[0]

        # initiate

        cfg = mmcv.Config.fromfile(
            'configs/apis/pose_estimator.cascade_rcnn+hrnet.yaml')
        video = mmcv.VideoReader('data/video/frame/cut_13.mp4')

        model = init_pose_estimator(**cfg, device=0)

        self.model.eval()

        if visualization:
            video_capture = cv2.VideoCapture(self.arg.video)
        else:
            video_capture = cv2.VideoCapture(path + filename)
        video_length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        pose_tracker = naive_pose_tracker(data_frame=video_length)

        # pose estimation
        start_time = time.time()
        frame_index = 0
        video = list()

        while(True):
            # get image
            ret, orig_image = video_capture.read()
            if orig_image is None:
                break
            source_H, source_W, _ = orig_image.shape
            orig_image = cv2.resize(
                orig_image, (256 * source_W // source_H, 256))
            H, W, _ = orig_image.shape
            video.append(orig_image)

            estimator_result = inference_pose_estimator(model, orig_image)

            # # pose estimation

            max_person_bbox_index = -1
            max_area = -1
            person_boxes = estimator_result['person_bbox']

            for index, person_box in enumerate(person_boxes):
                current_area = abs(
                    person_box[0] - person_box[2])*abs(person_box[1]-person_box[3])

                if current_area > max_area:
                    max_person_bbox_index = index

            keypoints = np.array([[np.array([p[0], p[1], round(s[0], 2)]) for p, s in zip(
                estimator_result['joint_preds'][max_person_bbox_index].round().astype(
                    int).tolist(),
                estimator_result['joint_scores'][max_person_bbox_index].tolist())]])

            multi_pose = keypoints
            if len(multi_pose.shape) != 3:
                continue

            # normalization
            multi_pose[:, :, 0] = multi_pose[:, :, 0]/W
            multi_pose[:, :, 1] = multi_pose[:, :, 1]/H
            multi_pose[:, :, 0:2] = multi_pose[:, :, 0:2] - 0.5
            multi_pose[:, :, 0][multi_pose[:, :, 2] == 0] = 0
            multi_pose[:, :, 1][multi_pose[:, :, 2] == 0] = 0

            # pose tracking
            pose_tracker.update(multi_pose, frame_index)
            frame_index += 1
            print('Pose estimation ({}/{}).'.format(frame_index, video_length))

        data_numpy = pose_tracker.get_skeleton_sequence()

        return video, data_numpy

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
                            default='../video/catch_wire_3.mp4',
                            help='Path to video')
        parser.add_argument('--openpose',
                            default='../../../../openpose/build',
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
        parser.set_defaults(
            config='./configs/recognition/st_gcn_aaai18/datax/demo_offline.yaml')
        parser.set_defaults(print_log=False)
        # endregion yapf: enable

        return parser


class naive_pose_tracker():
    """ A simple tracker for recording person poses and generating skeleton sequences.
    For actual occasion, I recommend you to implement a robuster tracker.
    Pull-requests are welcomed.
    """

    def __init__(self, data_frame=128, num_joint=17, max_frame_dis=np.inf):
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
                pad = current_frame-latest_frame-1
                new_trace = self.cat_pose(trace, p, pad, pad_mode)
                self.trace_info[matching_trace] = (new_trace, current_frame)

            else:
                new_trace = np.array([p])
                self.trace_info.append((new_trace, current_frame))

        self.latest_frame = current_frame

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
                coeff = [(p+1)/(pad+1) for p in range(pad)]
                interp_pose = [(1-c)*last_pose + c*pose for c in coeff]
                trace = np.concatenate((trace, interp_pose), 0)
        new_trace = np.concatenate((trace, [pose]), 0)
        return new_trace

    # calculate the distance between a existing trace and the input pose

    def get_dis(self, trace, pose):
        last_pose_xy = trace[-1, :, 0:2]
        curr_pose_xy = pose[:, 0:2]

        mean_dis = ((((last_pose_xy - curr_pose_xy)**2).sum(1))**0.5).mean()
        wh = last_pose_xy.max(0) - last_pose_xy.min(0)
        scale = (wh[0] * wh[1]) ** 0.5 + 0.0001
        is_close = mean_dis < scale * self.max_frame_dis
        return mean_dis, is_close
