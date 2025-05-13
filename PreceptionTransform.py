import cv2
import numpy as np


class PerspectiveTransformation:

    def __init__(self, scaling_factor):
        '''
        create warps for perception transformation
        one for front view
        one for top view
        '''
        self.src = np.float32([(423, 349),     # top-left
                               (16, 701),     # bottom-left
                               (1233, 697),    # bottom-right
                               (792, 342)])    # top-right

        self.dst = np.float32([(100, 0),
                               (100, 720),
                               (1100, 720),
                               (1100, 0)])

        self.src = self.src * scaling_factor
        self.dst = self.dst * scaling_factor

        self.top_warp = cv2.getPerspectiveTransform(self.src, self.dst)
        self.front_warp = cv2.getPerspectiveTransform(self.dst, self.src)

    def create_top_view(self, img):
        '''
        function returns the top view
        '''
        return cv2.warpPerspective(img, self.top_warp, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

    def create_front_view(self, img):
        '''
        function returns the front view
        '''
        return cv2.warpPerspective(img, self.front_warp, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
