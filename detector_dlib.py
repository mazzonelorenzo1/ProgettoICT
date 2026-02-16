import dlib

class DlibFaceDetector:
    def __init__(self, upsample: int = 1):
        # Initialize the standard HOG (Histogram of Oriented Gradients) face detector
        self.detector = dlib.get_frontal_face_detector()
        self.upsample = upsample

    def detect(self, bgr_img):
        # Run the detector on the BGR image
        # 'upsample' increases image resolution to detect smaller faces
        # The -1 argument requests scores for all detections

        # Returns: dlib rectangles (bounding boxes), detection scores, and sub-detector indices
        rects, scores, _ = self.detector.run(bgr_img, self.upsample, -1)
        return rects, scores
