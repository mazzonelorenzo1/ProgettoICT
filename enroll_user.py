import cv2
from face.detector_dlib import DlibFaceDetector
from face.align_dlib import DlibAligner5pt
from face.sphereface_embedder import SphereFaceEmbedder
from face.db import FaceDB
from face.verifier import FaceVerifier
from face.pipeline import FacePipeline

detector = DlibFaceDetector()
aligner = DlibAligner5pt("../assets/shape_predictor_5_face_landmarks.dat")
embedder = SphereFaceEmbedder("../model/sphere20a_20171020.pth")
db = FaceDB("../face_db.pkl")
verifier = FaceVerifier(db)

pipeline = FacePipeline(detector, aligner, embedder, verifier, db)

imgs = [
    cv2.imread("../data/calib/Filippo/Filippo_1.jpg"),
    cv2.imread("../data/calib/Filippo/Filippo_2.jpg"),
    cv2.imread("../data/calib/Filippo/Filippo_3.jpg"),
]

imgs = [img for img in imgs if img is not None]

res = pipeline.enroll_user("Filippo", imgs)
print(res)
