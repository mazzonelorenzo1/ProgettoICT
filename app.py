import base64

from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import os

from .detector_dlib import DlibFaceDetector
from .align_dlib import DlibAligner5pt
from .sphereface_embedder import SphereFaceEmbedder
from .db import FaceDB
from .verifier import FaceVerifier
from .pipeline import FacePipeline


# ----------------------------
# Flask app
# ----------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = Flask(__name__, template_folder=os.path.join(ROOT_DIR, "templates"))



# ----------------------------
# Key
# ----------------------------
def _require_key():
    expected = os.environ.get("PEEPHOLE_API_KEY", "")
    if not expected:
        return True

    got = request.headers.get("X-API-Key", "")
    return got == expected


# ----------------------------
# Build pipeline (run once)
# ----------------------------
def build_pipeline():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


    detector = DlibFaceDetector()
    aligner = DlibAligner5pt(
        os.path.join(BASE_DIR, "assets", "shape_predictor_5_face_landmarks.dat")
    )
    embedder = SphereFaceEmbedder(
        os.path.join(BASE_DIR, "model", "sphere20a_20171020.pth")
    )
    db = FaceDB(os.path.join(BASE_DIR, "face_db.pkl"))
    verifier = FaceVerifier(db)

    return FacePipeline(detector, aligner, embedder, verifier, db)


pipeline = build_pipeline()


def _b64_to_bgr_image(data_url: str):
    """
    data_url: stringa tipo 'data:image/jpeg;base64,...'
    ritorna immagine BGR OpenCV
    """
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    img_bytes = base64.b64decode(data_url)
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def home():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/identify")
def identify():
    if "image" not in request.files:
        return jsonify({"recognized": "no", "person": "Unknown", "error": "no_image"}), 400

    file = request.files["image"]
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"recognized": "no", "person": "Unknown", "error": "invalid_image"}), 400

    result = pipeline.identify(img)

    recognized = bool(result.get("ok", False))
    person = result.get("user") if recognized else "Unknown"

    return jsonify({
        "recognized": "yes" if recognized else "no",
        "person": person
    })


@app.post("/enroll")
def enroll():
    data = request.get_json(silent=True) or {}
    user = (data.get("user") or "").strip()
    images = data.get("images") or []

    if not _require_key():
        return jsonify({"ok": False, "error": "unauthorized"}), 401

    if not user:
        return jsonify({"ok": False, "error": "missing_user"}), 400
    if not isinstance(images, list) or len(images) == 0:
        return jsonify({"ok": False, "error": "missing_images"}), 400

    bgr_imgs = []
    for i, durl in enumerate(images):
        img = _b64_to_bgr_image(durl)
        if img is not None:
            bgr_imgs.append(img)

    if len(bgr_imgs) == 0:
        return jsonify({"ok": False, "error": "invalid_images"}), 400

    res = pipeline.enroll_user(user, bgr_imgs)

    # salva DB su disco
    pipeline.db.save()

    return jsonify(res)


# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)
