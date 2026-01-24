# src/face/detector_dlib.py
import dlib

class DlibFaceDetector:
    def __init__(self, upsample: int = 1):
        self.detector = dlib.get_frontal_face_detector()
        self.upsample = upsample

    def detect(self, bgr_img):
        # ritorna rectangles dlib
        rects, scores, _ = self.detector.run(bgr_img, self.upsample, -1)
        return rects, scores
