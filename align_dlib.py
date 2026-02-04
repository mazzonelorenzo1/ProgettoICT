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

    def _shape_to_3pts(self, shape) -> np.ndarray:
        # --- CORREZIONE SPECCHIO ---
        # Invertiamo la logica:
        # Se prima era specchiata, scambiamo eye_left con eye_right.

        # Punti 0-1: Occhio del soggetto a DX (Sinistra nell'immagine)
        # Punti 2-3: Occhio del soggetto a SX (Destra nell'immagine)

        # Calcoliamo i centroidi
        eye_img_left_x = (shape.part(0).x + shape.part(1).x) / 2.0
        eye_img_left_y = (shape.part(0).y + shape.part(1).y) / 2.0

        eye_img_right_x = (shape.part(2).x + shape.part(3).x) / 2.0
        eye_img_right_y = (shape.part(2).y + shape.part(3).y) / 2.0

        nose_x, nose_y = shape.part(4).x, shape.part(4).y

        # ORDINE CORRETTO per _REF_PTS standard:
        # Di solito è [Right_Eye_Img, Left_Eye_Img, Nose] oppure viceversa.
        # Se era specchiata, proviamo a scambiarli qui sotto:

        pts = np.array([
            [eye_img_right_x, eye_img_right_y],  # Prima coordinata (es. Ref point ~65.0)
            [eye_img_left_x, eye_img_left_y],  # Seconda coordinata (es. Ref point ~30.0)
            [nose_x, nose_y]
        ], dtype=np.float32)

        return pts

    def align(self, bgr_img, rect) -> np.ndarray:
        # Convertiamo in RGB per permettere a dlib di trovare i landmark
        rgb_input = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        shape = self.sp(rgb_input, rect)

        # Usa il metodo _shape_to_3pts corretto che abbiamo scritto prima
        src_pts = self._shape_to_3pts(shape)

        # Calcola trasformazione usando occhi e naso
        tfm = get_similarity_transform_for_cv2(src_pts, _REF_PTS[:3])

        # Restituiamo BGR (Standard OpenCV).
        # Niente "facce blu" se salvi questa immagine su disco.
        aligned_bgr = cv2.warpAffine(bgr_img, tfm, _CROP_SIZE)

        return aligned_bgr
