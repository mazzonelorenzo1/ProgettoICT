import cv2
import torch
from face.detector_dlib import DlibFaceDetector
from face.align_dlib import DlibAligner5pt
from face.sphereface_embedder import SphereFaceEmbedder
from face.db import FaceDB
from face.verifier import FaceVerifier
from face.pipeline import FacePipeline
from face.TelegramNotify import TelegramNotifier


def main():
    # 🔹 QUI entra l'immagine
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("Running on CPU")

    img = cv2.imread("data/Lorenzo_test_3.jpg")
    if img is None:
        raise RuntimeError("Immagine non letta")

    detector = DlibFaceDetector()
    aligner = DlibAligner5pt("assets/shape_predictor_5_face_landmarks.dat")
    embedder = SphereFaceEmbedder("model/sphere20a_20171020.pth", device = device)
    db = FaceDB("face_db.pkl")
    verifier = FaceVerifier(
        db,
        threshold=0.4,
        margin=0.0,
        mode="topk_mean",  # "best" = massimo su N foto
        default_max_samples=4,
        per_user_max_samples={"Lorenzo": 8}
    )

    notifier = TelegramNotifier(
        token="8555129415:AAF8FOdqbFxlxpLPYFZ0_gFzsxArx2QT_WQ",
        chat_id="-5105924827"
    )

    pipeline = FacePipeline(detector, aligner, embedder, verifier, db, notifier=notifier)

    result = pipeline.identify(img)
    print(result)

if __name__ == "__main__":
    main()
