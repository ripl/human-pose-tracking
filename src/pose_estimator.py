#!/usr/bin/env python3.6

import json
import os
from queue import Queue
from threading import Thread

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

SIZE20M = 20 * 1024 * 1024


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

        self.left_crop = 0.0
        self.right_crop = 0.0
        self.trt_expected_w = 224
        self.trt_expected_h = 224
        self.roi = None

        num_buffers = 2
        self._queue_used = Queue(num_buffers)
        self._queue_ready = Queue(num_buffers)
        self._estimator = Thread(target=self._estimate, daemon=True)
        for _ in range(num_buffers):
            self._queue_used.put(
                torch.zeros(
                    (3, self.trt_expected_h, self.trt_expected_w),
                    device=self.device,
                    dtype=torch.uint8
                )
            )
        self._input = torch.zeros(
            (3, self.trt_expected_h, self.trt_expected_w),
            device=self.device,
            dtype=torch.float
        )

        rospy.init_node('pose_estimator')
        rospy.Subscriber('/camera/color/image_raw',
                         Image,
                         callback=self.callback,
                         queue_size=1,
                         buff_size=SIZE20M)
        self.pub = rospy.Publisher('/estimated_poses',
                                   Int64MultiArray,
                                   queue_size=1)

        self._estimator.start()
        rospy.loginfo('Pose Estimator Node is Up!')
        rospy.spin()

    def _estimate(self):
        while not rospy.is_shutdown():
            buf_uint = self._queue_ready.get()
            buf = self._input.copy_(buf_uint).div_(255.0)

            # send to TRT pose
            # data = transforms.functional.to_tensor(img).to(self.device)
            buf.sub_(self.mean).div_(self.std)
            data = buf[None, ...]
            with torch.no_grad():
                cmap, paf = self.model_trt(data)
            cmap, paf = cmap.cpu(), paf.cpu()
            # topology: shape=[n_bones, 4]
            # counts: shape=[1]
            # objects: shape=[1, n_obj_candidates, n_kps(=18)]
            # peaks: shape=[1, n_kps(=18), n_kp_candidates, 2]
            counts, objects, peaks = self.parse_objects(cmap, paf)

            # free buffer
            self._queue_used.put(buf_uint)

            # pose_data: shape=[n_objs, n_kps(=18), 2]
            n_objs = counts.item()
            n_kps = objects.shape[2]
            poses = np.full((n_objs, n_kps, 2), -1, dtype=int)
            for i_obj in range(n_objs):
                for j_kp in range(n_kps):
                    k = objects[0][i_obj][j_kp]
                    if k < 0:
                        continue
                    p = peaks[0][j_kp][k]
                    poses[i_obj, j_kp] = [
                        self.roi[2] + (self.roi[3] - self.roi[2]) * p[1],
                        self.roi[0] + (self.roi[1] - self.roi[0]) * p[0]
                    ]

            self.pub.publish(np_bridge.to_multiarray_i64(poses))

    def callback(self, data):
        buf_uint = self._queue_used.get()
        img = self.cv_bridge.imgmsg_to_cv2(data, desired_encoding='rgb8')

        # perform crop some pixels off left and right (t, b, l, r)
        if self.roi is None:
            h, w, _ = img.shape
            t, l = 0, int((w - self.trt_expected_w) / 2)
            self.roi = (t, t + self.trt_expected_h, l, l + self.trt_expected_w)

        img = img[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
        buf_uint.copy_(torch.from_numpy(img.transpose(2, 0, 1)))
        self._queue_ready.put(buf_uint)


if __name__ == '__main__':
    PoseEstimator()
