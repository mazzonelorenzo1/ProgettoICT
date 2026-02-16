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
    # Select processing device (GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("Running on CPU")

    # Load the input image
    img = cv2.imread("data/calib/Lorenzo/Lorenzo_7.jpg")
    if img is None:
        raise RuntimeError("Image not read")

    # Initialize pipeline components
    detector = DlibFaceDetector()
    aligner = DlibAligner5pt("assets/shape_predictor_5_face_landmarks.dat")
    embedder = SphereFaceEmbedder("model/sphere20a_20171020.pth", device = device)
    db = FaceDB("face_db.pkl")

    # Configure verifier with validated thresholds
    verifier = FaceVerifier(
        db,
        threshold=0.26,
        margin=0.05,
        mode="topk_mean",  # Use average of best N matches
        default_max_samples=4,
        per_user_max_samples={"Lorenzo": 8}
    )

    # Setup Telegram bot for notifications
    notifier = TelegramNotifier(
        token="8555129415:AAF8FOdqbFxlxpLPYFZ0_gFzsxArx2QT_WQ",
        chat_id="-5105924827"
    )

    # Assemble and run the full pipeline
    pipeline = FacePipeline(detector, aligner, embedder, verifier, db, notifier=notifier)

    result = pipeline.identify(img)
    print(result)

if __name__ == "__main__":
    main()
