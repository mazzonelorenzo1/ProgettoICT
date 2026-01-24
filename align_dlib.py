# src/face/align_dlib.py
import numpy as np
import dlib
import cv2

from sphereface_pytorch.matlab_cp2tform import get_similarity_transform_for_cv2  # come nel notebook

_REF_PTS = np.array([
    [30.2946, 51.6963],
    [65.5318, 51.5014],
    [48.0252, 71.7366],
    [33.5493, 92.3655],
    [62.7299, 92.2041]
], dtype=np.float32)

_CROP_SIZE = (96, 112)  # (w, h) come warpAffine nel notebook

class DlibAligner5pt:
    def __init__(self, shape_predictor_path: str):
        self.sp = dlib.shape_predictor(shape_predictor_path)

    def _shape_to_5pts(self, shape) -> np.ndarray:
        # shape: dlib.full_object_detection con 5 landmarks
        pts = [(shape.part(i).x, shape.part(i).y) for i in range(5)]
        return np.array(pts, dtype=np.float32)

    def align(self, bgr_img, rect) -> np.ndarray:
        shape = self.sp(bgr_img, rect)
        src_pts = self._shape_to_5pts(shape)

        tfm = get_similarity_transform_for_cv2(src_pts, _REF_PTS)
        aligned = cv2.warpAffine(bgr_img, tfm, _CROP_SIZE)
        return aligned  # BGR aligned 96x112
