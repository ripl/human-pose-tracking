#!/usr/bin/env python

from dataclasses import dataclass

import numpy as np
import rospy
import tyro
from cv_bridge import CvBridge
from human_pose_tracking.pose_utils import pose_vis_highlight
from mmpose.apis import MMPoseInferencer
from np_bridge import to_ros_array
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray


@dataclass
class PoseEstimator:
    """Pose Estimator Node"""

    depth_masking: bool = False
    """Whether to apply depth-guided masking to the RGB image before pose detection."""
    depth_lo: int = 500
    """Depth lower bound of the depth-guided masking (measured in millimeters)."""
    depth_hi: int = 2500
    """Depth upper bound of the depth-guided masking (measured in millimeters)."""

    def main(self):
        self.inferencer = MMPoseInferencer(pose2d='human')
        self.pose_vis_highlight = pose_vis_highlight()
        self.cv_bridge = CvBridge()
        self.is_busy = False
        rospy.init_node('pose_estimator')
        rospy.Subscriber('/camera/color/image_raw', Image, callback=self.rgb_callback, queue_size=1, buff_size=1 << 23)
        if self.depth_masking:
            def depth_callback(data):
                self.depth = self.cv_bridge.imgmsg_to_cv2(data)
            rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, callback=depth_callback, queue_size=1, buff_size=1 << 23)
            self.depth = None
        self.pub = rospy.Publisher('/estimated_poses', Float64MultiArray, queue_size=1)
        self.pub_vis = rospy.Publisher('/estimated_poses_vis', Image, queue_size=1)
        rospy.loginfo('Pose Estimator Node is Up!')
        rospy.spin()

    def rgb_callback(self, data):
        if self.is_busy or self.depth_masking and self.depth is None:
            return
        self.is_busy = True
        img = self.cv_bridge.imgmsg_to_cv2(data).copy()
        if self.depth_masking:
            mask = np.logical_or(self.depth < self.depth_lo, self.depth > self.depth_hi)
            img[mask] = img[mask].mean(axis=0)
        pred_instances = next(self.inferencer(img, return_datasample=True))['predictions'][0].pred_instances
        self.is_busy = False
        self.pub.publish(to_ros_array(np.concatenate((pred_instances.keypoints, pred_instances.keypoint_scores[..., None]), axis=-1)))
        img = self.pose_vis_highlight._draw_instances_kpts(img, pred_instances)
        self.pub_vis.publish(self.cv_bridge.cv2_to_imgmsg(img[:, ::-1], encoding='rgb8', header=data.header))


if __name__ == '__main__':
    tyro.cli(PoseEstimator).main()
