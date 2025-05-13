from ObjectDetector import *
from PreceptionTransform import PerspectiveTransformation
from LaneLines import LaneLines
from Thresholding import Thresholding


class SelfDrivingVehicle:
    def __init__(self, first_frame, scaling_factor):
        self.thresholding = Thresholding()
        self.transform = PerspectiveTransformation(scaling_factor)
        self.lanelines = LaneLines()
        self.object_detector = ObjectDetection(first_frame, scaling_factor)
        self.frame = None
        self.scaling_factor = scaling_factor

    def forward(self, frame):
        self.frame = frame

        # ===== OBJECT DETECTION =====
        detected_img = self.object_detector.detect_objects(frame)
        obj_boxes = self.object_detector.last_detected_boxes()

        # Mask HUD region in detected obj view
        detected_img[0:int(270 * self.scaling_factor), 0:int(400 * self.scaling_factor)] = 0

        # ===== LANE DETECTION =====
        top_view = self.transform.create_top_view(frame)
        threshold_image = self.thresholding.lane_threshold(top_view)
        lane_image = self.lanelines.forward(threshold_image)

        # Generate red mask for object-lane overlap
        red_mask = self.object_roi_detection(top_view, obj_boxes, lane_image)

        # ===== APPLY MASK & MERGE LAYERS =====
        merged_top_view = self.apply_mask(red_mask, lane_image)
        front_view = self.transform.create_front_view(merged_top_view)

        if not np.array_equal(lane_image, threshold_image):
            out_img = cv2.addWeighted(front_view, 1, frame, 0.6, 0)
        else:
            out_img = frame

        out_img = self.lanelines.plot(out_img, self.scaling_factor)
        combined = cv2.addWeighted(detected_img, 1, out_img, 1, 0)

        return combined

    def apply_mask(self, red_mask, lane_image):
        if len(lane_image.shape) == 2:
            lane_image = cv2.cvtColor(lane_image, cv2.COLOR_GRAY2BGR)

        red_mask = cv2.resize(red_mask, (lane_image.shape[1], lane_image.shape[0]))
        if len(red_mask.shape) == 2:
            red_mask = cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR)

        return cv2.addWeighted(lane_image, 1, red_mask, 1, 0)

    def object_roi_detection(self, top_view, obj_boxes, lane_image):
        if len(lane_image.shape) == 3 and lane_image.shape[2] == 3:
            lane_mask = lane_image[:, :, 1] > 150
        else:
            lane_mask = np.zeros_like(lane_image, dtype=bool)

        red_mask = np.zeros_like(top_view)

        if obj_boxes and hasattr(self.lanelines, "left_fit") and hasattr(self.lanelines, "right_fit"):
            left_fit = self.lanelines.left_fit
            right_fit = self.lanelines.right_fit

            # Object bottom centers in front view
            bottom_centers = np.array([[x + w // 2, y + h] for (x, y, w, h) in obj_boxes], dtype='float32').reshape(-1, 1, 2)

            # Transform to top view
            transformed_centers = cv2.perspectiveTransform(bottom_centers, self.transform.top_warp)
            transformed_centers = transformed_centers.astype(int).reshape(-1, 2)

            for (x, y, w, h), (bx, by) in zip(obj_boxes, transformed_centers):
                if not (0 <= bx < self.frame.shape[1] and 0 <= by < self.frame.shape[0]):
                    continue
                if not lane_mask[by, bx]:
                    continue

                # Vertical red slice
                red_mask[:, max(bx - 5, 0):min(bx + 5, self.frame.shape[1])] = [0, 0, 255]

                # Highlight lane region below object
                y_coords = np.arange(by, self.frame.shape[0])
                left_x = (left_fit[0] * y_coords ** 2 + left_fit[1] * y_coords + left_fit[2]).astype(int)
                right_x = (right_fit[0] * y_coords ** 2 + right_fit[1] * y_coords + right_fit[2]).astype(int)

                left_x = np.clip(left_x, 0, self.frame.shape[1]-1)
                right_x = np.clip(right_x, 0, self.frame.shape[1]-1)

                for i, y_lane in enumerate(y_coords):
                    lx, rx = sorted([left_x[i], right_x[i]])
                    red_mask[y_lane, lx:rx] = [0, 0, 255]

        return red_mask




