#!/usr/bin/env python3.6

import json
import os
from queue import Queue
from threading import Thread

import cv2
import numpy as np
import rospy
import torch
import trt_pose.coco
from cv_bridge import CvBridge
from np_bridge import np_bridge
from sensor_msgs.msg import Image
from std_msgs.msg import Int64MultiArray
from torch2trt import TRTModule
from trt_pose.parse_objects import ParseObjects
from geometry_msgs.msg import Point, PointStamped


SIZE40M = 40 * 1024 * 1024


class PoseEstimator:
    def __init__(self) -> None:
        resources_path = os.path.join(os.environ['CPK_PROJECT_PATH'],
                                      'packages/trt_pose/resources')

        OPTIMIZED_MODEL = os.path.join(resources_path,
                                       'resnet18_baseline_att_224x224_A_epoch_249_trt.pth')
        self.model_trt = TRTModule()
        self.model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))

        self.device = torch.device('cuda')

        # from TRT Pose
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).to(self.device)[:, None, None]
        self.std = torch.Tensor([0.229, 0.224, 0.225]).to(self.device)[:, None, None]

        with open(os.path.join(resources_path, 'human_pose.json'), 'r') as f:
            human_pose = json.load(f)

        topology = trt_pose.coco.coco_category_to_topology(human_pose)
        self.parse_objects = ParseObjects(topology)

        self.cv_bridge = CvBridge()

        self.trt_expected_w = 224
        self.trt_expected_h = 224
        self.roi = None
        self.scale = None
        self.shift = None

        num_buffers = 2
        self._queue_used = Queue(num_buffers)
        self._queue_ready = Queue(num_buffers)
        self._estimator = Thread(target=self._estimate)
        for _ in range(num_buffers):
            self._queue_used.put(
                torch.empty(
                    (3, self.trt_expected_h, self.trt_expected_w),
                    device=self.device,
                    dtype=torch.uint8
                )
            )
        self._input = torch.empty(
            (3, self.trt_expected_h, self.trt_expected_w),
            device=self.device,
            dtype=torch.float
        )

        self._cmap = torch.empty(
            (1, 18, 56, 56),
            device=self.device,
            dtype=torch.float16
        )
        self._paf = torch.empty(
            (1, 42, 56, 56),
            device=self.device,
            dtype=torch.float16
        )

        self._cmap_cpu = torch.empty(
            (1, 18, 56, 56),
            dtype=torch.float32
        )
        self._paf_cpu = torch.empty(
            (1, 42, 56, 56),
            dtype=torch.float32
        )

        rospy.init_node('pose_estimator')
        rospy.Subscriber('/camera/color/image_raw',
                         Image,
                         callback=self.callback,
                         queue_size=1,
                         buff_size=SIZE40M)
        self.pub = rospy.Publisher('/estimated_poses',
                                   Int64MultiArray,
                                   queue_size=1)
        self.hpub = rospy.Publisher('/estimated_poses/header',
                                    PointStamped,
                                    queue_size=1)

        self._estimator.start()
        rospy.loginfo('Pose Estimator Node is Up!')
        rospy.spin()

    def callback(self, data):
        img = self.cv_bridge.imgmsg_to_cv2(data, desired_encoding='rgb8')

        if self.roi is None:
            h, w = img.shape[:2]
            self.roi = (0, h, 0, w)
            # self.roi = (0, 224, 0, 224)
            self.scale = np.array([(self.roi[1] - self.roi[0]),
                                   (self.roi[3] - self.roi[2])])
            self.shift = np.array([self.roi[0], self.roi[2]])

        img = cv2.resize(img, (self.trt_expected_w, self.trt_expected_h))
        # img = img[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]

        buf = self._queue_used.get()

        buf.copy_(torch.from_numpy(img.transpose(2, 0, 1)))

        self._queue_ready.put((data.header, buf))

    def _estimate(self):
        while not rospy.is_shutdown():
            header, buf = self._queue_ready.get()

            data = self._input.copy_(buf).div_(255.0)

            self._queue_used.put(buf)

            # send to TRT Pose
            data.sub_(self.mean).div_(self.std)
            with torch.no_grad():
                cmap, paf = self.model_trt(data[None, ...])

            self._cmap.copy_(cmap)
            self._paf.copy_(paf)

            self._cmap_cpu.copy_(self._cmap)
            self._paf_cpu.copy_(self._paf)

            # topology: shape=[n_bones, 4]
            # counts: shape=[1]
            # objects: shape=[1, n_obj_candidates, n_kps(=18)]
            # peaks: shape=[1, n_kps(=18), n_kp_candidates, 2]
            counts, objects, peaks = self.parse_objects(self._cmap_cpu, self._paf_cpu)

            peaks = (peaks * self.scale + self.shift).flip(-1)
            n_objs = counts.item()
            n_kps = objects.shape[2]
            objects = objects[0, :n_objs].long()
            poses = np.full((n_objs, n_kps, 2), -1, dtype=int)
            for i_obj in range(n_objs):
                ks = objects[i_obj]
                inds = torch.nonzero(ks >= 0).long()
                poses[i_obj, inds] = peaks[0, inds, ks[inds]]

            self.pub.publish(np_bridge.to_multiarray_i64(poses))
            # NOTE: this is only used so that we can do `rostopic delay`
            self.hpub.publish(PointStamped(header, Point(0, 0, 0)))


if __name__ == '__main__':
    PoseEstimator()
