import cv2

from face.detector_dlib import DlibFaceDetector
from face.align_dlib import DlibAligner5pt
from face.sphereface_embedder import SphereFaceEmbedder
from face.db import FaceDB
from face.verifier import FaceVerifier
from face.pipeline import FacePipeline

def main():
    # 🔹 QUI entra l'immagine
    img = cv2.imread("data/calib/Lorenzo/Lorenzo_5.jpg")
    if img is None:
        raise RuntimeError("Immagine non letta")

    detector = DlibFaceDetector()
    aligner = DlibAligner5pt("assets/shape_predictor_5_face_landmarks.dat")
    embedder = SphereFaceEmbedder("model/sphere20a_20171020.pth")
    db = FaceDB("face_db.pkl")
    verifier = FaceVerifier(
        db,
        threshold=0.51,
        margin=0.01,
        mode="topk_mean",  # "best" = massimo su N foto
        default_max_samples=4,
        per_user_max_samples={"Lorenzo": 8}
    )

    pipeline = FacePipeline(detector, aligner, embedder, verifier, db)

    result = pipeline.identify(img)
    print(result)

if __name__ == "__main__":
    main()
