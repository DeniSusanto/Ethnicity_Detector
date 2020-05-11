__author__ = 'Deni Susanto'
__email__ = 'densus@hku.hk'

import dlib
import imutils
from imutils import face_utils
import cv2
import numpy as np
import config

F_L_PREDICTOR_PATH = config.F_L_PREDICTOR_PATH
FL_WIDTH_RESIZE = config.FL_WIDTH_RESIZE

class FacialLandmark():
    #if landmark is not None, then assumed that the passed image already alligned
    def __init__(self, image, landmark = None):
        if not landmark:
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor(F_L_PREDICTOR_PATH)

            (ori_height, ori_width) = image.shape[:2]
            self.image_height, self.image_width = ori_height, ori_width
            resized_image = imutils.resize(image, width=FL_WIDTH_RESIZE)
            (trans_height, trans_width) = resized_image.shape[:2]

            ratio_x = ori_width/trans_width
            ratio_y = ori_height/trans_height

            gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 1)

            self.one_bb = False
            self.landmark_detected = False
            if len(rects) == 1:
                self.one_bb = True
                rect = rects[0]
                shape = predictor(gray, rect)
                if shape:
                    landmark = face_utils.shape_to_np(shape)
                    if len(landmark) == 68:
                        self.landmark_detected = True
                        adjusted_landmark = [(int(round(x * ratio_x)), int(round(y * ratio_y))) for (x,y) in landmark]
                        adjusted_landmark = np.asarray(adjusted_landmark)
                        (x, y, w, h) = face_utils.rect_to_bb(rect)
                        re, le = tuple(adjusted_landmark[36]), tuple(adjusted_landmark[45])
                        angle = self._get_angle_to_x_axis(re, le)
                        rotated_image = imutils.rotate(image, angle)
                        center = (ori_width // 2, ori_height // 2)
                        self.image = rotated_image
                        self.landmark = np.array([self._rotate_point(tuple(lm), center, angle) for lm in adjusted_landmark])
        else:
            self.one_bb = None
            self.landmark_detected = None
            self.image = image
            self.landmark = landmark
            (ori_height, ori_width) = image.shape[:2]
            self.image_height, self.image_width = ori_height, ori_width
                    
    def _returnImage(self, x1, x2, y1, y2, highLight):
        if not highLight:
            return self.image[y1:y2, x1:x2]
        else:
            img_copy = self.image.copy()
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 1)
            return img_copy
        
    def _rotate_point(self, point, center, angle):
        rad = np.radians(360-angle)
        p_x, p_y = point
        c_x, c_y = center
        v_x = p_x - c_x
        v_y = p_y - c_y

        p_r_x = (v_x * np.cos(rad)) - (v_y * np.sin(rad))
        p_r_y = (v_x * np.sin(rad)) + (v_y * np.cos(rad))

        f_x = p_r_x + c_x
        f_y = p_r_y + c_y

        return [int(round(f_x)),int(round(f_y))]
    def _get_angle_to_x_axis(self, p1, p2):
        p1_x, p1_y = p1
        p2_x, p2_y = p2
        if p2_x != p1_x:
            m = (p2_y - p1_y) / (p2_x - p1_x)
        else:
            m = (p2_y - p1_y) / (p2_x - p1_x + np.finfo(float).eps)

        return np.degrees(np.arctan(m))

    def get_full_face_only(self, highlight = False):
        face_y_min = int((min(self.landmark[:,1]).flatten()[0]))
        face_y_max = int(np.asarray(max(self.landmark[:,1]).flatten()[0]))
        face_height = face_y_max - face_y_min 
        forehead_height = int(face_height * 0.25)
        
        x1 = self.landmark[0][0]
        x2 = self.landmark[16][0]
        y1 = max((face_y_min - forehead_height), 0)
        y2 = face_y_max
        
        if x1 <= 0 or x2 <=0 or x1 > self.image_width or x2 > self.image_width or y2 > self.image_height:
            return np.array([])
        
        return self._returnImage(x1, x2, y1, y2, highlight)        