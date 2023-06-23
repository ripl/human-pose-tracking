import numpy as np
from scipy.optimize import linear_sum_assignment


class KeypointsTracker:
    def __init__(self, depth_lo, depth_hi, timeout):
        self.depth_lo = depth_lo
        self.depth_hi = depth_hi
        self.timeout = timeout
        self.next_obj_id = 0
        self.objs = {}

    def register(self, kpts, t):
        # [kpts, spawn_time, last_seen]
        self.objs[self.next_obj_id] = [kpts, t, t]
        self.next_obj_id += 1

    def unregister(self, obj_id):
        del self.objs[obj_id]

    def update(self, kpts_lst, depth_lst, t):
        """
        Args:
            kpts_lst: (n_objs, n_kpts, n_dims)
            depths: (n_objs, n_kpts)
        """
        if self.objs:
            obj_ids, obj_vals = zip(*self.objs.items())
            D = np.linalg.norm(kpts_lst[:, None] - np.array([obj_val[0] for obj_val in obj_vals]), axis=-1).mean(axis=-1)
        else:
            D = np.empty((len(kpts_lst), 0))
        rows, cols = linear_sum_assignment(D)
        m = None
        for row, col in zip(rows, cols):
            obj = self.objs[obj_ids[col]]
            obj[0] = kpts_lst[row]
            obj[2] = t
            if self.depth_lo < depth_lst[row] < self.depth_hi and (m is None or obj[1] < m):
                highlight = row
                m = obj[1]
        should_highlight = m is None
        for row in set(range(D.shape[0])) - set(rows):
            self.register(kpts_lst[row], t)
            depth = depth_lst[row]
            if should_highlight and self.depth_lo < depth < self.depth_hi and (m is None or depth < m):
                highlight = row
                m = depth
        for col in set(range(D.shape[1])) - set(cols):
            obj_id = obj_ids[col]
            if t - self.objs[obj_id][2] > self.timeout:
                self.unregister(obj_id)
        return -1 if m is None else highlight
