#!/usr/bin/env python3.6

import json
import os

import cv2
import numpy as np
import rospy
import trt_pose.coco
from cv_bridge import CvBridge
from np_bridge import np_bridge
from sensor_msgs.msg import Image
from std_msgs.msg import Int64MultiArray


class PoseVisualizer:
    def __init__(self) -> None:
        resources_path = os.path.join(os.environ['CPK_PROJECT_PATH'], 'packages/trt_pose/resources')
        with open(os.path.join(resources_path, 'human_pose.json'), 'r') as f:
            human_pose = json.load(f)
        self.topology = trt_pose.coco.coco_category_to_topology(human_pose)

        self.img = None
        self.cv_bridge = CvBridge()
        rospy.init_node('pose_visualizer')
        rospy.Subscriber('/camera/color/image_raw', Image, callback=self.img_callback, queue_size=1)
        rospy.Subscriber('/tracked_poses', Int64MultiArray, callback=self.pose_callback, queue_size=10)
        self.pub = rospy.Publisher('/robot/xdisplay', Image, queue_size=1)
        rospy.loginfo('Pose Visualizer Node is Up!')
        rospy.spin()

    def img_callback(self, data):
        self.img = self.cv_bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')

    def pose_callback(self, data):
        poses = np_bridge.to_numpy_i64(data)
        img = self.img.copy()
        for i_obj in range(poses.shape[0]):
            color = (0, 255, 0) if poses[i_obj, -1, 0] else (128, 128, 128)
            for j_kp in range(poses.shape[1] - 1):
                x, y = poses[i_obj, j_kp]
                if x >= 0 and y >= 0:
                    cv2.circle(img, (x, y), 20, color, -1)
            for k in range(self.topology.shape[0]):
                c_a = self.topology[k][2]
                c_b = self.topology[k][3]
                if np.all(poses[i_obj][c_a] >= 0) and np.all(poses[i_obj][c_b] >= 0):
                    x0, y0 = poses[i_obj, c_a]
                    x1, y1 = poses[i_obj, c_b]
                    cv2.line(img, (x0, y0), (x1, y1), color, 10)
        img_msg = self.cv_bridge.cv2_to_imgmsg(img[:, ::-1], encoding='bgr8')
        self.pub.publish(img_msg)


if __name__ == '__main__':
    PoseVisualizer()
