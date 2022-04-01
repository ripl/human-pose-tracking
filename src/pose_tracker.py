#!/usr/bin/env python3.6

from time import time

import numpy as np
import rospy
from cv_bridge import CvBridge
from np_bridge import np_bridge
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from sensor_msgs.msg import Image
from std_msgs.msg import Int64MultiArray

SIZE20M = 20 * 1024 * 1024


class CentroidTracker:
    def __init__(self, depth_th=5000, timeout=3):
        self.depth_th = depth_th
        self.timeout = timeout
        self.nextObjectID = 0
        self.objects = {}

    def register(self, centroid, t):
        # [centroid, spawn_time, last_seen]
        self.objects[self.nextObjectID] = [centroid, t, t]
        self.nextObjectID += 1

    def unregister(self, objectID):
        del self.objects[objectID]

    def update(self, centroids, depths):
        t = time()
        if not len(centroids):
            for objectID in list(self.objects.keys()):
                if t - self.objects[objectID][2] > self.timeout:
                    self.unregister(objectID)
            return -1
        if not self.objects:
            for centroid in centroids:
                self.register(centroid, t)
            return 0
        objectIDs, objects = zip(*self.objects.items())
        D = distance.cdist(np.array([val[0] for val in objects]), centroids)
        rows, cols = linear_sum_assignment(D)
        m = None
        for row, col in zip(rows, cols):
            obj = self.objects[objectIDs[row]]
            obj[0] = centroids[col]
            obj[2] = t
            if depths[col] < self.depth_th and (m is None or obj[1] < m):
                k = col
                m = obj[1]
        for row in set(range(D.shape[0])).difference(set(rows)):
            objectID = objectIDs[row]
            if t - self.objects[objectID][2] > self.timeout:
                self.unregister(objectID)
        for col in set(range(D.shape[1])).difference(set(cols)):
            self.register(centroids[col], t)
        return k


class PoseTracker:
    def __init__(self) -> None:
        self.depth = None
        self.cv_bridge = CvBridge()
        self.tracker = CentroidTracker()
        rospy.init_node('pose_tracker')
        rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, callback=self.depth_callback, queue_size=1, buff_size=SIZE20M)
        rospy.Subscriber('/estimated_poses', Int64MultiArray, callback=self.pose_callback, queue_size=1)
        self.pub_pose = rospy.Publisher('/tracked_poses', Int64MultiArray, queue_size=1)
        # self.pub_depth = rospy.Publisher('filtered_depth', Image, queue_size=1)
        rospy.loginfo('Pose Tracker Node is Up!')
        rospy.spin()

    def depth_callback(self, data):
        self.depth = self.cv_bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')

    def pose_callback(self, data):
        # if self.depth is None:
        #     return
        poses = np_bridge.to_numpy_i64(data)
        centroids = poses[:, -1]
        is_valid = np.all(centroids >= 0, axis=1)
        centroids = centroids[is_valid]
        valid_id2id = {}
        n_objs = is_valid.shape[0]
        valid_id = 0
        for id in range(n_objs):
            if is_valid[id]:
                valid_id2id[valid_id] = id
                valid_id += 1
        # depths = [self.depth[c[1], c[0]] for c in centroids]
        depths = [1000] * len(centroids)
        k = self.tracker.update(centroids, depths)
        h = np.zeros((n_objs, 1, 2), dtype=int)
        if k >= 0:
            id = valid_id2id[k]
            h[id, 0] = 1
        tracked_poses = np.concatenate((poses, h), axis=1)
        self.pub_pose.publish(np_bridge.to_multiarray_i64(tracked_poses))


if __name__ == '__main__':
    PoseTracker()
