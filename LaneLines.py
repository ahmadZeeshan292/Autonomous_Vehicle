import cv2
import numpy as np


def hist(img):
    bottom_half = img[img.shape[0] // 2:, :]
    return np.sum(bottom_half, axis=0)


class LaneLines:

    def __init__(self):

        self.left_fit = None
        self.right_fit = None
        self.binary = None
        self.nonzero = None
        self.nonzerox = None
        self.nonzeroy = None
        self.dir = []

        # HYPERPARAMETERS
        # Number of sliding windows
        self.nwindows = 9
        # Width of the windows +/- margin
        self.margin = 100
        # Minimum number of pixels found to recenter window
        self.minpix = 50

    def forward(self, img):

        # print(f"left_fit = {self.left_fit}\n right_fit = {self.right_fit}")
        results = None
        self.extract_features(img)
        try:
            results = self.fit_poly(img)
        except Exception as e:
            return img

        return results


    def pixels_in_window(self, center, margin, height):
        """ Return all pixel that in a specific window

            pixelx (np.array): x coordinates of pixels that lie inside the window
            pixely (np.array): y coordinates of pixels that lie inside the window
        """
        topleft = (center[0] - margin, center[1] - height // 2)
        bottomright = (center[0] + margin, center[1] + height // 2)

        condx = (topleft[0] <= self.nonzerox) & (self.nonzerox <= bottomright[0])
        condy = (topleft[1] <= self.nonzeroy) & (self.nonzeroy <= bottomright[1])
        return self.nonzerox[condx & condy], self.nonzeroy[condx & condy]

    def extract_features(self, img):
        """ Extract features from a binary image

        Parameters:
            img (np.array): A binary image
        """
        self.img = img
        # Height of windows - based on nwindows and image shape
        self.window_height = img.shape[0] // self.nwindows

        # Identify the x and y positions of all nonzero pixel in the image
        self.nonzero = img.nonzero()
        self.nonzerox = np.array(self.nonzero[1])
        self.nonzeroy = np.array(self.nonzero[0])

    def find_lane_pixels(self, img):
        """Find lane pixels from a binary warped image.

        Parameters:
            img (np.array): A binary warped image

        Returns:
            leftx (np.array): x coordinates of left lane pixels
            lefty (np.array): y coordinates of left lane pixels
            rightx (np.array): x coordinates of right lane pixels
            righty (np.array): y coordinates of right lane pixels
            out_img (np.array): A RGB image that use to display result later on.
        """
        assert (len(img.shape) == 2)

        # Create an output image to draw on and visualize the result
        out_img = np.dstack((img, img, img))

        histogram = hist(img)
        midpoint = histogram.shape[0] // 2
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Current position to be update later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base
        y_current = img.shape[0] + self.window_height // 2

        # Create empty lists to reveice left and right lane pixel
        leftx, lefty, rightx, righty = [], [], [], []

        # Step through the windows one by one
        for _ in range(self.nwindows):
            y_current -= self.window_height
            center_left = (leftx_current, y_current)
            center_right = (rightx_current, y_current)

            good_left_x, good_left_y = self.pixels_in_window(center_left, self.margin, self.window_height)
            good_right_x, good_right_y = self.pixels_in_window(center_right, self.margin, self.window_height)

            # Append these indices to the lists
            leftx.extend(good_left_x)
            lefty.extend(good_left_y)
            rightx.extend(good_right_x)
            righty.extend(good_right_y)

            if len(good_left_x) > self.minpix:
                leftx_current = np.int32(np.mean(good_left_x))
            if len(good_right_x) > self.minpix:
                rightx_current = np.int32(np.mean(good_right_x))

        return leftx, lefty, rightx, righty, out_img

    def fit_poly(self, img):
        """
        Find the lane line from an image and draw it.
        """

        leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(img)

        if len(lefty) > 1500:
            self.left_fit = np.polyfit(lefty, leftx, 2)
        if len(righty) > 1500:
            self.right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        maxy = img.shape[0] - 1
        miny = img.shape[0] // 3
        if len(lefty):
            maxy = max(maxy, np.max(lefty))
            miny = min(miny, np.min(lefty))

        if len(righty):
            maxy = max(maxy, np.max(righty))
            miny = min(miny, np.min(righty))

        ploty = np.linspace(miny, maxy, img.shape[0])

        left_fitx = self.left_fit[0] * ploty ** 2 + self.left_fit[1] * ploty + self.left_fit[2]
        right_fitx = self.right_fit[0] * ploty ** 2 + self.right_fit[1] * ploty + self.right_fit[2]

        # Visualization
        for i, y in enumerate(ploty):
            l = int(left_fitx[i])
            r = int(right_fitx[i])
            y = int(y)
            cv2.line(out_img, (l, y), (r, y), (0, 255, 0))

        return out_img

    def plot(self, out_img, scaling_factor):
        np.set_printoptions(precision=6, suppress=True)

        # Initialize message variable
        msg = None
        object_pos = None
        try:
            lR, rR, pos = self.measure_curvature()
            object_pos = self.object_pos(out_img, pos)
        except Exception as e:
            msg = "No lane detected"
            lR, rR, pos = None, None, None

        if lR is not None and rR is not None:
            if abs(self.left_fit[0]) > abs(self.right_fit[0]):
                value = self.left_fit[0]
            else:
                value = self.right_fit[0]

            if abs(value) <= 0.00015:
                self.dir.append('F')
            elif value < 0:
                self.dir.append('L')
            else:
                self.dir.append('R')

        if len(self.dir) > 10:
            self.dir.pop(0)

        # Draw widget on top-left corner
        W = int(400 * scaling_factor)
        H = int(270 * scaling_factor)
        widget = np.copy(out_img[:H, :W])
        widget //= 2
        widget[0, :] = [0, 0, 255]
        widget[-1, :] = [0, 0, 255]
        widget[:, 0] = [0, 0, 255]
        widget[:, -1] = [0, 0, 255]
        out_img[:H, :W] = widget

        text_base_y = int(50 * scaling_factor)
        line_spacing = int(40 * scaling_factor)

        curvature_msg = None
        if msg is None:
            direction = max(set(self.dir), key=self.dir.count)
            curvature_msg = "Curvature = {:.0f} m".format(min(lR, rR)) if lR and rR else "Curvature = N/A"

            if direction == 'L':
                msg = "Left Curve Ahead"
            elif direction == 'R':
                msg = "Right Curve Ahead"
            elif direction == 'F':
                msg = "Keep Straight Ahead"

        cv2.putText(out_img, msg, org=(10, text_base_y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1 * scaling_factor,
                    color=(255, 255, 255), thickness=2)

        if lR and rR:
            cv2.putText(out_img, curvature_msg, org=(10, text_base_y + line_spacing),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1 * scaling_factor,
                        color=(255, 255, 255), thickness=2)

            if object_pos is not None and int(object_pos) <= 1:
                cv2.putText(out_img, "Very close object detected",
                            org=(10, text_base_y + 3 * line_spacing),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8 * scaling_factor,
                            color=(0, 255, 0), thickness=2)

            else:
                cv2.putText(out_img, "Good Lane Keeping",
                            org=(10, text_base_y + 3 * line_spacing),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.2 * scaling_factor,
                            color=(0, 255, 0), thickness=2)

            cv2.putText(out_img, "Vehicle is {:.2f} m away from center".format(pos),
                        org=(10, text_base_y + 4 * line_spacing),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.66 * scaling_factor,
                        color=(255, 255, 255), thickness=2)

            if object_pos is not None:
                cv2.putText(out_img, "Vehicle is {:.2f} m away from object".format(object_pos),
                            org=(10, text_base_y + 5 * line_spacing),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.66 * scaling_factor,
                            color=(255, 255, 255), thickness=2)

        return out_img

    def measure_curvature(self):
        '''
        This function computes the radius of the curvature
        R = (1 + (2Ay + B)^2)^1.5/ |2A|
        where pos calculates how far the vehicle is from the
        centre, the ym and xm are meters per pixel
        '''

        ym = 30 / self.img.shape[0]
        xm = 3.7 / 700

        left_fit = self.left_fit.copy()
        right_fit = self.right_fit.copy()
        y_eval = 700 * ym

        # Compute R_curve (radius of curvature)
        left_curveR = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
        right_curveR = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])

        xl = np.dot(self.left_fit, [700 ** 2, 700, 1])
        xr = np.dot(self.right_fit, [700 ** 2, 700, 1])
        pos = (self.img.shape[1] // 2 - (xl + xr) // 2) * xm
        return left_curveR, right_curveR, pos

    def object_pos(self, img, pos):
        # Define yellow range in HSV
        lower_yellow = np.array([15, 100, 100])
        upper_yellow = np.array([30, 255, 255])

        # Extract last N rows (e.g., last 50 rows)
        last_rows = img[:, :, :]

        # Create a mask for yellow in the last rows
        yellow_mask = cv2.inRange(last_rows, lower_yellow, upper_yellow)

        if cv2.countNonZero(yellow_mask) > 0:
            print("Yellow color detected in the last rows")

        ys, xs = np.where(yellow_mask > 0)

        if len(xs) > 0:
            closest_x = np.min(xs)
            furthest_x = np.max(xs)
            return min(abs(pos - closest_x), abs(pos - furthest_x)) * (3.7 / 700)
        else:
            return None
