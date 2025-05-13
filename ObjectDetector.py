import cv2
import numpy as np
from sklearn.cluster import DBSCAN

'''
about this class:
The class uses 3 main algorithms
1. DBSCAN
2. Lucas-Kanade optical flow
3. Shi Tomasi Corner Detection

Working in code:
i)   Apply Shi Tomasi algorithm to detect strong corners points to track in initial frame
ii)  Apply Lucas-Kanade Algorithm for estimating hwo these points move in next frame
iii) Cluster the motion vectors via DBSCAN to detect object-level motion

Theory behind Algorithm:
Shi Tomasi Algorithm (relevant corner point detection):
1. Select an appropiate window size
2. Determine which windows produce very large variations in intensities when moved along gradients 
3. Compute an R score when appropiate window found
4. Apply threshold to this score and important corners are selected and marked

Lucas-Kanade (optical flow estimation method):
The algorithm assumes the following between the current and previous frame:
1. Brightness is constant
2. Small motion enough to approximate via first order taylor expansion
3. small region move together: pixel's neighbors have the same Δ along x and y

The algorithm works by:
1. Selecting an window 
2. Place the corner points in that window for the prev frame
3. Estimate the Δ (u, v) of this whole window between the frames
4. linearize brightness change via 1st order taylor expansion
5. compute the u, v for each window via least square 

DBSCAN(Density-Based Spatial Clustering of Application with Noise):
An clustering algorithm especially useful for spatial or motion data

Parameters
The Algorithm groups data based on how closely packed data points are
its required two parameters:
1. Maximum distance to consider a neighbor(eps in code)
2. Minimum number of neighbours to form a core point (min_samples in code)
what the above means:
say what should be pixel difference (eps) along x and y to classify min_samples as neighbours 

Working:
Label every point in dataset as one of three types,
based on how many points are within eps distance
Depending on the count:
1. Core point: ≥ min_samples 
2. Border Point: < min_samples, its neighbour is an core point
3. noise: Not a core point, not in any core's neighbour

i) Start with a unvisited core point
ii) Create a new cluster and add all its eps-neighbours to the cluster
iii) for each neighbour if core point, add its neighbours too (recursive expansion)
iv) repeat until no more desity-connected points can be added
v) border points are added if fall within eps neighborhood of a core point
vi) noise is discarded


How the three algorithms work:
1. The relevant corner points are detected from Shi Tomosi algorithm
2. The points are input to the Lucas kanade algorithm where an motion vector is 
computed for each point in some window
3. The motion vectors are feed to DBSCAN to detect moving objects where noise which
are random/unstable movements that are discarded
'''


class ObjectDetection:
    def __init__(self, first_frame, scaling_factor):
        first_frame = first_frame.copy()
        self.old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        self.feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self.p0 = cv2.goodFeaturesToTrack(self.old_gray, mask=None, **self.feature_params)
        self.mask = np.zeros_like(first_frame)
        self.frame_count = 0
        self.next_object_id = 0
        self.object_id_map = {}
        self.cur_frame = None
        self.boundary_box = None
        self.scaling_factor = scaling_factor

    def apply_optical_flow_mask(self, cur_frame):
        self.cur_frame = cur_frame
        cur_gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)

        if self.p0 is None:
            return cur_frame, None

        p1, st, _ = cv2.calcOpticalFlowPyrLK(self.old_gray, cur_gray, self.p0, None, **self.lk_params)
        good_new = p1[st == 1]
        good_old = self.p0[st == 1]

        # draw motion trails
        for new, old in zip(good_new, good_old):
            a, b = new.ravel()
            c, d = old.ravel()
            self.mask = cv2.line(self.mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)

        points = good_new
        db = DBSCAN(eps=int(30*self.scaling_factor), min_samples=10).fit(points)  # cluster nearby points

        return cv2.add(self.cur_frame, self.mask), (db, points, cur_gray, good_new)

    def merge_boxes(self, bboxes, iou_threshold=0.3, distance_thresh=50):
        merged = []
        used = [False] * len(bboxes)

        def boxes_overlap(b1, b2):
            x1, y1, w1, h1 = b1
            x2, y2, w2, h2 = b2
            return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)

        def boxes_close(b1, b2, thresh):
            x1, y1, w1, h1 = b1
            x2, y2, w2, h2 = b2
            cx1, cy1 = x1 + w1 // 2, y1 + h1 // 2
            cx2, cy2 = x2 + w2 // 2, y2 + h2 // 2
            return np.linalg.norm([cx1 - cx2, cy1 - cy2]) < thresh

        for i in range(len(bboxes)):
            if used[i]:
                continue
            x1, y1, w1, h1 = bboxes[i]
            merged_box = [x1, y1, x1 + w1, y1 + h1]
            used[i] = True
            for j in range(i + 1, len(bboxes)):
                if used[j]:
                    continue
                b2 = bboxes[j]
                if boxes_overlap(bboxes[i], b2) or boxes_close(bboxes[i], b2, distance_thresh):
                    x2, y2, w2, h2 = b2
                    merged_box[0] = min(merged_box[0], x2)
                    merged_box[1] = min(merged_box[1], y2)
                    merged_box[2] = max(merged_box[2], x2 + w2)
                    merged_box[3] = max(merged_box[3], y2 + h2)
                    used[j] = True
            x, y, x2, y2 = merged_box
            merged.append((x, y, x2 - x, y2 - y))
        return merged

    def trace_objects(self, cur_frame):
        img, result = self.apply_optical_flow_mask(cur_frame)
        if result is None:
            return np.zeros_like(cur_frame)  # return pure black image if no detections

        db, points, cur_gray, good_new = result
        labels = db.labels_
        boxes = []

        h, w = cur_frame.shape[:2]
        dx, dy = int(w * 0.0), int(h * 0.0)  # 10% margin from each side

        for label in set(labels):
            if label == -1:
                continue  # noise
            cluster_points = points[labels == label]
            x, y, w_box, h_box = cv2.boundingRect(cluster_points.astype(np.int32))

            # Only accept boxes entirely within the central region
            if dx <= x and x + w_box <= w - dx and dy <= y and y + h_box <= h - dy:
                boxes.append((x, y, w_box, h_box))

        merged_boxes = self.merge_boxes(boxes)
        self.boundary_box = merged_boxes

        # Create black background and mask the edges
        black_image = np.zeros_like(cur_frame)
        central_mask = np.zeros_like(cur_frame[:, :, 0])
        central_mask[dy:h - dy, dx:w - dx] = 1
        central_mask_3ch = np.stack([central_mask]*3, axis=-1)

        for box in merged_boxes:
            x, y, w_box, h_box = box
            obj_id = self.next_object_id
            self.next_object_id += 1
            cv2.rectangle(black_image, (x, y), (x + w_box, y + h_box), (255, 0, 0), 2)
            cv2.putText(black_image, f"Object {obj_id}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6 * self.scaling_factor, (0, 255, 255), 2)

        # Mask out everything except the central region
        black_image = cv2.bitwise_and(black_image, black_image, mask=central_mask)

        self.old_gray = cur_gray.copy()
        self.p0 = good_new.reshape(-1, 1, 2)

        self.frame_count += 1
        if self.frame_count % 5 == 0:
            new_points = cv2.goodFeaturesToTrack(cur_gray, mask=None, **self.feature_params)
            if new_points is not None:
                if self.p0 is not None:
                    self.p0 = np.concatenate((self.p0, new_points), axis=0)
                else:
                    self.p0 = new_points

        if self.frame_count % 10000 == 0:
            self.p0 = cv2.goodFeaturesToTrack(cur_gray, mask=None, **self.feature_params)
            self.old_gray = cur_gray.copy()
            self.mask = np.zeros_like(cur_frame)

        return black_image

    def detect_objects(self, cur_frame):
        cur_frame = cur_frame.copy()
        return self.trace_objects(cur_frame)

    def last_detected_boxes(self):
        return self.boundary_box
