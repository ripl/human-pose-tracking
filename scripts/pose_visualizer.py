#!/usr/bin/env python3

from dataclasses import dataclass

import cv2
import numpy as np
import rospy
import tyro
from cv_bridge import CvBridge
from human_pose_tracking.msg import TrackedPoses
from human_pose_tracking.pose_utils import pose_vis, pose_vis_highlight
from mmengine.structures import InstanceData
from np_bridge import to_np_array
from sensor_msgs.msg import Image


@dataclass
class PoseVisualizer:
    """Pose Visualizer Node"""

    dev_w: int = 1024
    """Width of the display device."""
    dev_h: int = 600
    """Height of the display device."""

    def main(self):
        self.pose_vis = pose_vis()
        self.pose_vis_highlight = pose_vis_highlight()
        self.img = None
        self.cv_bridge = CvBridge()
        rospy.init_node('pose_visualizer')
        rospy.Subscriber('/camera/color/image_raw', Image, callback=self.img_callback, queue_size=1, buff_size=1 << 23)
        rospy.Subscriber('/tracked_poses', TrackedPoses, callback=self.pose_callback, queue_size=1)
        self.pub = rospy.Publisher('/robot/xdisplay', Image, queue_size=1)
        rospy.loginfo('Pose Visualizer Node is Up!')
        rospy.spin()

    def img_callback(self, data):
        if self.img is None:
            r = data.width / data.height
            if r < self.dev_w / self.dev_h:
                self.tgt_h = None
                self.tgt_w = round(self.dev_h * r)
            else:
                self.tgt_h = round(self.dev_w / r)
                self.tgt_w = None
        self.img = self.cv_bridge.imgmsg_to_cv2(data)

    def pose_callback(self, data):
        if self.img is None:
            return
        poses = to_np_array(data.poses)
        instances = InstanceData(keypoints=poses[..., :2], keypoint_scores=poses[..., 2])
        highlight_mask = np.arange(len(instances)) == data.highlight
        img = self.pose_vis._draw_instances_kpts(self.img, instances[~highlight_mask])
        img = self.pose_vis_highlight._draw_instances_kpts(img, instances[highlight_mask])
        img_mean = self.img.mean(axis=(0, 1))
        img_padded = np.broadcast_to(img_mean, (self.dev_h, self.dev_w, len(img_mean))).astype(np.uint8)
        if self.tgt_h:
            img_padded[(self.dev_h - self.tgt_h) // 2:(self.dev_h + self.tgt_h) // 2] = cv2.resize(img, (self.dev_w, self.tgt_h), interpolation=cv2.INTER_CUBIC)
        else:
            img_padded[:, (self.dev_w - self.tgt_w) // 2:(self.dev_w + self.tgt_w) // 2] = cv2.resize(img, (self.tgt_w, self.dev_h), interpolation=cv2.INTER_CUBIC)
        self.pub.publish(self.cv_bridge.cv2_to_imgmsg(img_padded[:, ::-1, ::-1], encoding='bgr8'))


if __name__ == '__main__':
    tyro.cli(PoseVisualizer).main()
