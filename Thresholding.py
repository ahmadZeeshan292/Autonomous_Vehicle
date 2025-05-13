import cv2
import numpy as np


class Thresholding:
    def __init__(self):
        pass

    def lane_threshold(self, image):

        '''
        Things to consider if the yellow and white lanes are not
        prominent apply morphological operators or in this case narrow
        the range of white and yellow shades to remove noise and to
        distinguish the minimum shades closest to yellow and white
        as possible
        '''

        '''
        this function captures the essence of white and yellow lanes by
        taking relative and absolute threshold wrt to lightness, saturation
        hue and value
        '''

        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        h = hls[:, :, 0]
        l = hls[:, :, 1]
        s = hls[:, :, 2]
        v = hsv[:, :, 2]

        '''
        white line are very bright meaning they have high lightness l 
        but have low saturation s thus the values 200, 255 are chosen
        based on lightness and 0 and 60 are chosen based on saturation
        '''

        white_mask = self.absolute_threshold(l, 220, 255)  # stricter lightness
        white_mask &= self.absolute_threshold(s, 0, 60)

        '''
        yellow lines are have hue between of about 25 thus and absolute 
        threshold of 15, 35 is used to capture these shades and in terms of 
        saturation and lightness these are very bright and colorful thus and 
        value of 0.5 and 1 wrt l and s was used
        '''

        yellow_mask = self.absolute_threshold(h, 20, 30)  # narrower yellow hue band
        yellow_mask &= self.relative_threshold(s, 0.5, 1.0)
        yellow_mask &= self.relative_threshold(v, 0.5, 1.0)

        '''
        combine the masks created for yellow and while lanes
        '''

        combined = cv2.bitwise_or(white_mask, yellow_mask)

        return yellow_mask

    def relative_threshold(self, image, lo, hi):
        vmin = np.min(image)
        vmax = np.max(image)

        vlo = vmin + (vmax - vmin) * lo
        vhi = vmax + (vmax - vmin) * hi

        return np.uint8((image >= vlo) & (image <= vhi)) * 255

    def absolute_threshold(self, image, lo, hi):
        return np.uint8((image >= lo) & (image <= hi)) * 255