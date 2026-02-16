import numpy as np
import dlib
import cv2

from sphereface_pytorch.matlab_cp2tform import get_similarity_transform_for_cv2  # as in the notebook

_REF_PTS = np.array([
    [30.2946, 51.6963],
    [65.5318, 51.5014],
    [48.0252, 71.7366],
    [33.5493, 92.3655],
    [62.7299, 92.2041]
], dtype=np.float32)

_CROP_SIZE = (96, 112)


class DlibAligner5pt:
    def __init__(self, shape_predictor_path: str):
        self.sp = dlib.shape_predictor(shape_predictor_path)

    def _shape_to_3pts(self, shape) -> np.ndarray:

        # Points 0-1: Subject's Right Eye (Left in the image)
        # Points 2-3: Subject's Left Eye (Right in the image)

        # Calculate centroids
        eye_img_left_x = (shape.part(0).x + shape.part(1).x) / 2.0
        eye_img_left_y = (shape.part(0).y + shape.part(1).y) / 2.0

        eye_img_right_x = (shape.part(2).x + shape.part(3).x) / 2.0
        eye_img_right_y = (shape.part(2).y + shape.part(3).y) / 2.0

        nose_x, nose_y = shape.part(4).x, shape.part(4).y

        # correct for standard _REF_PTS:
        # Usually [Right_Eye_Img, Left_Eye_Img, Nose] or vice versa.
        # If mirrored, try swapping them below:

        pts = np.array([
            [eye_img_right_x, eye_img_right_y],  # First coordinate
            [eye_img_left_x, eye_img_left_y],  # Second coordinate
            [nose_x, nose_y]
        ], dtype=np.float32)

        return pts

    def align(self, bgr_img, rect) -> np.ndarray:
        # Convert to RGB to allow dlib to find landmarks
        rgb_input = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        shape = self.sp(rgb_input, rect)

        # Use the correct _shape_to_3pts method defined earlier
        src_pts = self._shape_to_3pts(shape)

        # Calculate transformation using eyes and nose
        tfm = get_similarity_transform_for_cv2(src_pts, _REF_PTS[:3].copy())

        # Return BGR (Standard OpenCV).
        aligned_bgr = cv2.warpAffine(bgr_img, tfm, _CROP_SIZE)

        return aligned_bgr
