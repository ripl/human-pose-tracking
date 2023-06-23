#!/usr/bin/env python3

from dataclasses import dataclass

import numpy as np
import rospy
import tyro
from cv_bridge import CvBridge
from human_pose_tracking.keypoints_tracker import KeypointsTracker
from human_pose_tracking.msg import TrackedPoses
from np_bridge import to_np_array, to_ros_array
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray


@dataclass
class PoseTracker:
    """Pose Tracker Node"""

    score_th: float = 0.7
    """Score threshold for a keypoint to be considered valid."""
    valid_th: float = 0.7
    """Valid keypoint percentage threshold for a pose to be considered valid."""
    timeout: float = 1
    """Timeout for a pose to be considered lost (measured in seconds)."""
    depth_lo: int = 1000
    """Depth lower bound for a pose to be highlighted (measured in millimeters)."""
    depth_hi: int = 2000
    """Depth upper bound for a pose to be highlighted (measured in millimeters)."""

    def main(self):
        self.tracker = KeypointsTracker(depth_lo=self.depth_lo, depth_hi=self.depth_hi, timeout=self.timeout)
        self.kids_validation = (0, 5, 6, 7, 8, 9, 10)
        self.kids_tracking = (0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16)
        self.depth = None
        self.cv_bridge = CvBridge()
        rospy.init_node('pose_tracker')
        rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, callback=self.depth_callback, queue_size=1, buff_size=1 << 23)
        rospy.Subscriber('/estimated_poses', Float64MultiArray, callback=self.pose_callback, queue_size=1)
        self.pub = rospy.Publisher('/tracked_poses', TrackedPoses, queue_size=1)
        rospy.loginfo('Pose Tracker Node is Up!')
        rospy.spin()

    def depth_callback(self, data):
        self.depth = self.cv_bridge.imgmsg_to_cv2(data)

    def pose_callback(self, data):
        if self.depth is None:
            return
        h, w = self.depth.shape
        poses = to_np_array(data)
        poses = poses[(poses[:, self.kids_validation, 2] > self.score_th).mean(axis=-1) > self.valid_th]
        n_poses, n_kpts = poses.shape[:-1]
        depths = np.zeros((n_poses, n_kpts), dtype=np.uint16)
        for i in range(n_poses):
            for j in range(n_kpts):
                x, y = poses[i, j, :2].round().astype(int)
                if 0 <= x < w and 0 <= y < h and poses[i, j, 2] > self.score_th:
                    depths[i, j] = self.depth[y, x]
        highlight = self.tracker.update(poses[:, self.kids_tracking, :2], [np.median(depth[depth > 0]) for depth in depths[:, self.kids_tracking]], rospy.get_time())
        self.pub.publish(TrackedPoses(poses=to_ros_array(poses), depths=to_ros_array(depths), highlight=highlight))


if __name__ == '__main__':
    tyro.cli(PoseTracker).main()
