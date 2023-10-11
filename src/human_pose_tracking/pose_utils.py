import numpy as np
from mmpose.visualization.local_visualizer import PoseLocalVisualizer

kid2name = {0: 'nose', 1: 'left_eye', 2: 'right_eye', 3: 'left_ear', 4: 'right_ear', 5: 'left_shoulder', 6: 'right_shoulder', 7: 'left_elbow', 8: 'right_elbow', 9: 'left_wrist', 10: 'right_wrist', 11: 'left_hip', 12: 'right_hip', 13: 'left_knee', 14: 'right_knee', 15: 'left_ankle', 16: 'right_ankle'}

skeleton_cfg = ((15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6))

keypoints_visible = np.array([False] * 5 + [True] * 12)


def pose_vis():
    return PoseLocalVisualizer(
        kpt_color=np.full((17, 3), 150),
        link_color=np.full((19, 3), 100),
        skeleton=skeleton_cfg,
        line_width=6,
        radius=9
    )


def pose_vis_highlight():
    return PoseLocalVisualizer(
        kpt_color=np.array(((51, 153, 255),
                            (51, 153, 255),
                            (51, 153, 255),
                            (51, 153, 255),
                            (51, 153, 255),
                            (0, 255, 0),
                            (255, 128, 0),
                            (0, 255, 0),
                            (255, 128, 0),
                            (0, 255, 0),
                            (255, 128, 0),
                            (0, 255, 0),
                            (255, 128, 0),
                            (0, 255, 0),
                            (255, 128, 0),
                            (0, 255, 0),
                            (255, 128, 0))),
        link_color=np.array(((0, 255, 0),
                             (0, 255, 0),
                             (255, 128, 0),
                             (255, 128, 0),
                             (51, 153, 255),
                             (51, 153, 255),
                             (51, 153, 255),
                             (51, 153, 255),
                             (0, 255, 0),
                             (255, 128, 0),
                             (0, 255, 0),
                             (255, 128, 0),
                             (51, 153, 255),
                             (51, 153, 255),
                             (51, 153, 255),
                             (51, 153, 255),
                             (51, 153, 255),
                             (51, 153, 255),
                             (51, 153, 255))),
        skeleton=skeleton_cfg,
        line_width=12,
        radius=18,
    )
