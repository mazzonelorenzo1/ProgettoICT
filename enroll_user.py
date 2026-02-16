import cv2
from face.detector_dlib import DlibFaceDetector
from face.align_dlib import DlibAligner5pt
from face.sphereface_embedder import SphereFaceEmbedder
from face.db import FaceDB
from face.verifier import FaceVerifier
from face.pipeline import FacePipeline

# Initialize individual components
detector = DlibFaceDetector()
aligner = DlibAligner5pt("../assets/shape_predictor_5_face_landmarks.dat")
embedder = SphereFaceEmbedder("../model/sphere20a_20171020.pth")
db = FaceDB("../face_db.pkl")
verifier = FaceVerifier(db)

# Assemble the full processing pipeline
pipeline = FacePipeline(detector, aligner, embedder, verifier, db)

# Load images for enrollment
imgs = [
    cv2.imread("../data/calib/Alessandro/Ale_1.jpg"),
    cv2.imread("../data/calib/Alessandro/Ale_2.jpg"),
    cv2.imread("../data/calib/Alessandro/Ale_3.jpg"),
]

# Filter out any images that failed to load
imgs = [img for img in imgs if img is not None]

# Register the user in the database
res = pipeline.enroll_user("Alessandro", imgs)
print(res)
