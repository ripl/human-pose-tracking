import numpy as np
from scipy.optimize import linear_sum_assignment


class KeypointsTracker:
    def __init__(self, depth_lo, depth_hi, border_ratio, timeout_tracking, timeout_highlight):
        self.depth_lo = depth_lo
        self.depth_hi = depth_hi
        self.border_ratio = border_ratio
        self.timeout_tracking = timeout_tracking
        self.timeout_highlight = timeout_highlight
        self.next_obj_id = 0
        self.objs = {}

    def set_img_width(self, img_w):
        self.border_lo = np.round(img_w * self.border_ratio).astype(int)
        self.border_hi = img_w - self.border_lo

    def register(self, kpts, t):
        # [kpts, last_seen, last_highlight]
        obj = [kpts, t, None]
        self.objs[self.next_obj_id] = obj
        self.next_obj_id += 1
        return obj

    def unregister(self, obj_id):
        del self.objs[obj_id]

    def update(self, kpts_lst, depth_lst, t):
        # kpts_lst: (n_objs, n_kpts, n_dims)
        # depths: (n_objs, n_kpts)
        if self.objs:
            obj_ids, obj_vals = zip(*self.objs.items())
            D = np.linalg.norm(kpts_lst[:, None] - np.array([obj_val[0] for obj_val in obj_vals]), axis=-1).mean(axis=-1)
        else:
            D = np.empty((len(kpts_lst), 0))
        rows, cols = linear_sum_assignment(D)
        earliest_highlight = None
        min_depth = None
        highlight = -1
        hightlight_obj = None
        for row, col in zip(rows, cols):
            obj = self.objs[obj_ids[col]]
            obj[0] = kpts_lst[row]
            obj[1] = t
            if self.depth_lo < depth_lst[row] < self.depth_hi and self.border_lo <= np.median(kpts_lst[row, :, 0]) < self.border_hi:
                if obj[2] is not None and t - obj[2] < self.timeout_highlight and (earliest_highlight is None or obj[2] < earliest_highlight):
                    highlight = row
                    hightlight_obj = obj
                    earliest_highlight = obj[2]
                elif earliest_highlight is None and (min_depth is None or depth_lst[row] < min_depth):
                    highlight = row
                    hightlight_obj = obj
                    min_depth = depth_lst[row]
        if earliest_highlight is not None:
            for obj in self.objs.values():
                if obj[2] is not None and obj[2] > earliest_highlight:
                    obj[2] = None
        should_highlight = highlight == -1
        for row in set(range(D.shape[0])) - set(rows):
            obj = self.register(kpts_lst[row], t)
            depth = depth_lst[row]
            if should_highlight and self.depth_lo < depth < self.depth_hi and (min_depth is None or depth < min_depth) and self.border_lo <= np.median(kpts_lst[row, :, 0]) < self.border_hi:
                highlight = row
                hightlight_obj = obj
                min_depth = depth
        if hightlight_obj is not None:
            hightlight_obj[2] = t
        for col in set(range(D.shape[1])) - set(cols):
            obj_id = obj_ids[col]
            if t - self.objs[obj_id][1] > self.timeout_tracking:
                self.unregister(obj_id)
        return highlight
